from typing import List
from src.models import FunctionDefinition, UserPrompt


class PromptBuilder:
    """
        Class that generates a complete prompt in string format
        for the LLM use.
    """

    def prompt_build(self,
                     data_prompt: UserPrompt,
                     functions: List[FunctionDefinition]
                     ) -> str:
        """
        generates a complete prompt in string format,
        including instructions, available functions, and constraints.
        This string is intended to be passed
        to the `encode()` function of the `llm_sdk`.
        """
        tools_desc = ""
        for f in functions:
            func_desc: str = f"- Name: {f.name}\n  Description: {f.description}\n  Parameters: {f.parameters}\n\n"
            tools_desc += func_desc

        system_prompt = (
            "### INSTRUCTIONS\n"
            "You are a specialized function-calling engine. Your ONLY task is "
            "to map user queries to specific JSON tool calls.\n"
            "STRICT RULE: You must output valid JSON and NOTHING ELSE."
            "No thinking, no intro, no outro.\n\n"
            "### AVAILABLE FUNCTIONS\n"
            f"{tools_desc}\n"
            "### OUTPUT FORMAT\n"
            "{\n"
            '  "prompt": "The exact user query",\n'
            '  "name": "function_name",\n'
            '  "parameters": {"param_name": value}\n'
            "}\n\n"
            "### CRITICAL CONSTRAINTS\n"
            "- Never include <think> tags.\n"
            "- Never explain your choice."
        )

        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{data_prompt.prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
