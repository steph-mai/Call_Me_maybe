import pytest
import json
from src.load import Loader

# --- Tests de succès ---


def test_load_valid_json(tmp_path):
    """Vérifie le chargement normal d'un fichier valide."""
    d = tmp_path / "test.json"
    content = {"query": "Hello"}
    d.write_text(json.dumps(content))

    loader = Loader()
    assert loader.load_file(str(d)) == content

# --- Tests d'erreurs ---


def test_load_missing_file():
    """Vérifie l'erreur quand le fichier n'existe pas."""
    loader = Loader()
    with pytest.raises(FileNotFoundError) as excinfo:
        loader.load_file("missing_file.json")
    assert "not found" in str(excinfo.value)


def test_load_invalid_json_content(tmp_path):
    """Vérifie l'erreur quand le contenu JSON est mal formé."""
    d = tmp_path / "corrupted.json"
    d.write_text('{ "prompt": "test" ')
    loader = Loader()
    with pytest.raises(ValueError) as excinfo:
        loader.load_file(str(d))
    assert "Invalid JSON" in str(excinfo.value)


def test_load_empty_file(tmp_path):
    """Vérifie l'erreur quand le fichier est totalement vide."""
    d = tmp_path / "empty.json"
    d.write_text("")

    loader = Loader()
    with pytest.raises(ValueError):
        loader.load_file(str(d))


def test_load_permission_denied(tmp_path):
    """Vérifie l'erreur quand le fichier est protégé en lecture."""
    d = tmp_path / "protected.json"
    d.write_text('{"secret": "data"}')
    d.chmod(0o000)

    loader = Loader()
    with pytest.raises(PermissionError):
        loader.load_file(str(d))

    d.chmod(0o644)
