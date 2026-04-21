import json
from pathlib import Path
from typing import Any
from src.models import FunctionDefinition, UserPrompt


class Loader:
    def load_file(self, file_path: str) -> Any:
        """
        Loads a JSON file.
        Handles cases: unsupported file type, missing file, permission error,
        invalid JSON, empty file.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Missing File: '{path.name}' not found")

        if path.suffix.lower() != ".json":
            raise ValueError(
                f"Unsupported file type: '{path.suffix}'. "
                f"Please provide a .json file"
                )

        if path.is_dir():
            raise IsADirectoryError("Expected a file, but found a directory")

        try:
            with open(file_path, mode="r", encoding="utf-8") as f:
                content = f.read().strip()
        except PermissionError:
            raise PermissionError(
                f"You don't have permisssion for this file: {file_path}"
                )

        if not content:
            raise ValueError(f"File is empty: '{path.name}'")

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{path.name}': "
                             f"Line {e.lineno}, column {e.colno}.")

    def get_functions(self, file_path: str) -> list[FunctionDefinition]:
        """
        Loads, parses, and validates function definitions from a JSON file.
        Returns a list of validated FunctionDefinition Pydantic objects
        """
        raw_functions = self.load_file(file_path)

        if not isinstance(raw_functions, list):
            raise ValueError(
                f"There is no valid list of functions in {file_path} "
                )

        functions = [FunctionDefinition(**fn) for fn in raw_functions]

        if not functions:
            raise ValueError(
                f"There is no valid list of functions in {file_path} "
                )

        return functions

    def get_prompts(self, file_path: str) -> list[UserPrompt]:
        """
        Loads, parses, and validates prompts from a JSON file.
        Returns a list of validated UserPrompt Pydantic objects
        """
        raw_prompts = self.load_file(file_path)

        if not isinstance(raw_prompts, list):
            raise ValueError(
                f"There is no valid list of prompts in {file_path} ")

        prompts = [UserPrompt(**pr) for pr in raw_prompts]

        if not prompts:
            raise ValueError(
                f"There is no valid list of prompts in {file_path} ")

        return prompts
