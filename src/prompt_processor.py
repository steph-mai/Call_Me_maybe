import sys
import json
from typing import List
from src.models import FunctionDefinition, FunctionCallResult


class PromptProcessor:
    """
    Orchestrates the high-level workflow of turning a user prompt
    into a validated FunctionCallResult.
    """
    def __init__(self, decoder):
        self.decoder = decoder

    def process(self,
                user_prompt: str,
                functions: List[FunctionDefinition],
                system_prompt_ids: List[int]
                ) -> FunctionCallResult:

        name_selection_ids = system_prompt_ids + self.decoder.llm.encode(
            f"\nQuery: {user_prompt}\nFunction:"
            )[0].tolist()
        selected_name = self.decoder.force_name(name_selection_ids, functions)

        selected_function_def = None
        for f in functions:
            if f.name == selected_name:
                selected_function_def = f
                break
        if selected_function_def is None:
            raise ValueError(
                f"Error: The function '{selected_name}' does not exist."
                )

        final_params: dict[str, float | int | bool | str] = {}
        for p_name, p_info in selected_function_def.parameters.items():
            sys.stdout.write(f" | {p_name}: ")

            instruction = f"\nTask: {user_prompt}\nFunction: {selected_name}\n{p_name}="
            if p_info.type == "string":
                instruction += '"'

            full_prompt_ids = (
                system_prompt_ids
                + self.decoder.llm.encode(instruction)[0].tolist()
            )

            raw_value = self.decoder.extract_param_value(
                full_prompt_ids,
                p_info.type)

            if isinstance(raw_value, str):
                raw_value = raw_value.replace('"', '').strip()
                if p_name == "regex":
                    while raw_value.count('(') > raw_value.count(')'):
                        raw_value += ')'
                    while raw_value.count('[') > raw_value.count(']'):
                        raw_value += ']'
                if p_name == "replacement" and "**" in raw_value:
                    raw_value = "*"

            final_params[p_name] = raw_value

        return FunctionCallResult(
            prompt=user_prompt,
            name=selected_name,
            parameters=final_params
            )
