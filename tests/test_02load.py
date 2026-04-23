import pytest
import json
from src.load import Loader

# --- SUCCESS TESTS ---


def test_load_valid_json(tmp_path) -> None:
    """Ensure the loading of a valid file."""
    d = tmp_path / "test.json"
    content = {"query": "Hello"}
    d.write_text(json.dumps(content))

    loader = Loader()
    assert loader.load_file(str(d)) == content

# --- FAILURE TESTS ---


def test_load_missing_file() -> None:
    """Ensure an error is raised if file not found."""
    loader = Loader()
    with pytest.raises(FileNotFoundError) as excinfo:
        loader.load_file("missing_file.json")
    assert "not found" in str(excinfo.value)


def test_load_invalid_json_content(tmp_path) -> None:
    """Ensure an error is raised with invalid json."""
    d = tmp_path / "corrupted.json"
    d.write_text('{ "prompt": "test" ')
    loader = Loader()
    with pytest.raises(ValueError) as excinfo:
        loader.load_file(str(d))
    assert "Invalid JSON" in str(excinfo.value)


def test_load_empty_file(tmp_path) -> None:
    """Verify that en empty file is rejected."""
    d = tmp_path / "empty.json"
    d.write_text("")

    loader = Loader()
    with pytest.raises(ValueError):
        loader.load_file(str(d))


def test_load_permission_denied(tmp_path) -> None:
    """Ensure an error is raised if there is no permission file."""
    d = tmp_path / "protected.json"
    d.write_text('{"secret": "data"}')
    d.chmod(0o000)

    loader = Loader()
    with pytest.raises(PermissionError):
        loader.load_file(str(d))

    d.chmod(0o644)
