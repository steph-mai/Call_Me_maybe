import json
from pathlib import Path
from typing import Any


class Loader:
    def load_file(self, file_path: str) -> Any:
        """
        Loads a JSON file. 
        Handles cases: unsupported file type, missing file, permission error, invalid JSON, empty file.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Missing File: '{path.name}' not found in: {path.absolute().parent}")

        if path.suffix.lower() != ".json":
            raise ValueError(f"Unsupported file type: '{path.suffix}'. Please provide a .json file")

        if path.is_dir():
            raise IsADirectoryError("Expected a file, but found a directory")

        try:
            content = path.read_text(encoding="utf-8").strip()
        except PermissionError:
            raise PermissionError(f"You don't have permisssion for this file: {file_path}")

        if not content:
            raise ValueError(f"File is empty: '{path.name}'")

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{path.name}': Line {e.lineno}, column {e.colno}.")
