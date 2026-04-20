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
        # On injecte le décodeur pour avoir accès à ses méthodes de force_name et extract_value
        self.decoder = decoder

    def process(self, prompt: str, functions: List[FunctionDefinition], static_ids: List[int]) -> FunctionCallResult:
        # --- 1. SÉLECTION DU NOM ---
        name_ids = static_ids + self.decoder.llm.encode(f"\nQuery: {prompt}\nFunction:")[0].tolist()
        selected_name = self.decoder._force_name(name_ids, functions)
        
        fn_def = next(f for f in functions if f.name == selected_name)

        final_params = {}
        for p_name, p_info in fn_def.parameters.items():
            sys.stdout.write(f" | {p_name}: ")

            instruction = f"\nTask: {prompt}\nFunction: {selected_name}\n{p_name}="
            if p_info.type == "string": 
                instruction += '"'

            p_ids = static_ids + self.decoder.llm.encode(instruction)[0].tolist()
            val = self.decoder._extract_value(p_ids, p_info.type)

            if isinstance(val, str):
                val = val.replace('"', '').strip()
                if p_name == "regex":
                    while val.count('(') > val.count(')'): val += ')'
                    while val.count('[') > val.count(']'): val += ']'
                if p_name == "replacement" and "**" in val:
                    val = "*"

            final_params[p_name] = val

        return FunctionCallResult(prompt=prompt, name=selected_name, parameters=final_params)