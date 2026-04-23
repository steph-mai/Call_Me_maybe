import sys
import json
from pathlib import Path
from typing import List, Any, Dict, Optional
from src.models import FunctionDefinition, FunctionCallResult, UserPrompt
from src.constrained_decoder import ConstrainedDecoder


class PromptProcessor:
    """
    Manages the process of converting a user prompt into a validated
    function call, using constrained decoding.
    """
    def __init__(self, decoder: ConstrainedDecoder) -> None:
        self.decoder = decoder
        # Cache to store results and avoid redundant LLM calls
        self._cache: Dict[str, FunctionCallResult] = {}

    def run(
            self,
            
            functions: List[FunctionDefinition],
            user_prompts: List[UserPrompt],
            output_path: str | Path
            ) -> None:
        """
        Main entry point: processes a list of prompts and saves results
        to a JSON file.
        Includes caching logic to skip already processed prompts.
        """
        # Convert Pydantic models to standard dictionaries
        # using the pydantic method 'model_dump'
        tools_list = [f.model_dump() for f in functions]

        system_prompt: str = (
            f"System: Tool Extractor. "
            f"Tools: {json.dumps(tools_list)}"
        )

        # Tokenize the system prompt containing available tools
        system_prompt_ids = self.decoder.llm.encode(
            system_prompt
            )[0].tolist()

        results: list[FunctionCallResult] = []

        for index, user_prompt in enumerate(user_prompts):
            prompt_text = user_prompt.prompt

            # Cache check
            if prompt_text in self._cache:
                result = self._cache[prompt_text]
                sys.stdout.write(
                    f"\n[{index+1}] Prompt: {prompt_text[:30]}... (CACHED)"
                    )
                results.append(result)
                continue

            sys.stdout.write(f"\n[{index+1}/{len(user_prompts)}] "
                             f"Prompt: {user_prompt.prompt[:80]}...")

            res: FunctionCallResult = self.process(
                user_prompt.prompt,
                functions,
                system_prompt_ids
            )

            # Store result in cache
            self._cache[prompt_text] = res

            results.append(res)

        # Save results to file
        output_data = [r.model_dump() for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

    def process(self,
                user_prompt: str,
                functions: List[FunctionDefinition],
                system_prompt_ids: List[int]
                ) -> FunctionCallResult:
        """
        Executes the function calling pipeline:
        1. Identifies the function name.
        2. Iteratively extracts each required parameter.
        3. Validates and cleans output via advanced recovery.
        """

        query_prompt = f"\nQuery: {user_prompt}\nFunction:"

        name_selection_ids = system_prompt_ids + self.decoder.llm.encode(
            query_prompt
            )[0].tolist()

        selected_name = self.decoder.force_name(name_selection_ids, functions)

        sys.stdout.write(f"\n  [NAME]: \033[94m{selected_name}\033[0m")

        # Retrieve the matching function definition
        selected_function_def: Optional[FunctionDefinition] = None
        for f in functions:
            if f.name == selected_name:
                selected_function_def = f
                break
        if selected_function_def is None:
            raise ValueError(
                f"Error: The function '{selected_name}' does not exist."
                )

        final_params: dict[str, float | int | bool | str] = {}

        # Iterative parameter extraction
        for p_name, p_info in selected_function_def.parameters.items():

            instruction = (
                f"\nTask: {user_prompt}\n"
                f"Function: {selected_name}\n"
                f"{p_name}="
            )

            # Pre-append a quote for string types to guide the model
            if p_info.type == "string":
                instruction += '"'

            full_prompt_ids = (
                system_prompt_ids
                + self.decoder.llm.encode(instruction)[0].tolist()
            )

            # Constrained extraction based on parameter type
            raw_value = self.decoder.extract_param_value(
                full_prompt_ids,
                p_info.type)

            # Post-processing and error recovery
            final_value = self._advanced_recovery(
                p_name, p_info.type,
                raw_value)

            final_params[p_name] = final_value

            if p_info.type == "number":
                color = "\033[92m"
            elif p_info.type == "boolean":
                color = "\033[95m"
            else:
                color = "\033[93m"
            sys.stdout.write(f" | {p_name}: {color}{final_value}\033[0m")
            sys.stdout.flush()

        return FunctionCallResult(
            prompt=user_prompt,
            name=selected_name,
            parameters=final_params
            )

    def _advanced_recovery(
            self, p_name: str, p_type: str, raw_value: Any
            ) -> Any:
        """Centralizes cleaning and repair mechanisms"""

        if p_type == "boolean":
            if str(raw_value).strip().lower() in (
                ["true", "1", "yes", "y", "t"]
            ):
                return True
            return False

        if isinstance(raw_value, str):
            val = raw_value.replace('"', '').strip()

            if p_name == "regex":
                for opening, closing in [('(', ')'), ('[', ']')]:
                    while val.count(opening) > val.count(closing):
                        val += closing

            if "**" in val:
                val = val.replace("**", "*")

            return val

        if p_type == "number":
            try:
                return float(raw_value)
            except (ValueError, TypeError):
                return 0.0

        return raw_value
