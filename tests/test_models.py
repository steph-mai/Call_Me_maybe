import pytest
from pydantic import ValidationError
from src.models import FunctionDefinition, UserPrompt, FunctionCallResult

# =================================================================
# 1. INPUT TESTS (Validation de function_definitions.json)
# =================================================================
class TestFunctionDefinition:

    def test_valid_function_definition(self):
        """Vérifie qu'une définition complète et correcte est acceptée."""
        data = {
            "name": "fn_add",
            "description": "Additionne deux nombres",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "returns": {"type": "number"}
        }
        func = FunctionDefinition(**data)
        assert func.name == "fn_add"
        assert func.parameters["a"].type == "number"

    @pytest.mark.parametrize("missing_field", ["name", "description", "parameters", "returns"])
    def test_missing_fields(self, missing_field):
        """Vérifie que chaque champ obligatoire déclenche une erreur s'il manque."""
        base_data = {
            "name": "fn_add",
            "description": "Adds two numbers",
            "parameters": {"a": {"type": "number"}},
            "returns": {"type": "number"}
        }
        del base_data[missing_field]
        with pytest.raises(ValidationError):
            FunctionDefinition(**base_data)

    def test_completely_empty(self):
        """Vérifie qu'un dictionnaire vide {} échoue."""
        with pytest.raises(ValidationError):
            FunctionDefinition(**{})

    def test_no_parameters(self):
        """Vérifie qu'une fonction sans paramètres est valide."""
        data = {
            "name": "fn_get_now",
            "description": "Get current time",
            "parameters": {},
            "returns": {"type": "string"}
        }
        func = FunctionDefinition(**data)
        assert func.parameters == {}

    def test_parameters_is_none(self):
        """Vérifie que 'parameters' ne peut pas être null/None."""
        data = {
            "name": "fn_test", "description": "d",
            "parameters": None, "returns": {"type": "s"}
        }
        with pytest.raises(ValidationError):
            FunctionDefinition(**data)

    def test_empty_name(self):
        """Vérifie que le nom de la fonction n'est pas une chaîne vide."""
        with pytest.raises(ValidationError):
            FunctionDefinition(name="", description="d", parameters={}, returns={"type": "s"})

    def test_invalid_return_format(self):
        """Vérifie que 'returns' est bien un objet avec une clé 'type'."""
        with pytest.raises(ValidationError):
            FunctionDefinition(name="f", description="d", parameters={}, returns="number")

    def test_parameter_type_unsupported(self):
        """Vérifie que seuls 'number', 'string' et 'boolean' sont acceptés."""
        data = {
            "name": "fn_invalid", "description": "d",
            "parameters": {"arg1": {"type": "array"}},
            "returns": {"type": "s"}
        }
        with pytest.raises(ValidationError):
            FunctionDefinition(**data)

    def test_parameter_type_not_string(self):
        """Vérifie que le type du paramètre n'est pas un nombre ou autre."""
        data = {
            "name": "fn_test", "description": "d",
            "parameters": {"arg1": {"type": 123}},
            "returns": {"type": "s"}
        }
        with pytest.raises(ValidationError):
            FunctionDefinition(**data)


# =================================================================
# 2. PROMPTS TESTS (Validation de function_calling_tests.json)
# =================================================================
class TestInputPrompts:

    def test_valid_prompt(self):
        """Vérifie qu'un prompt normal est accepté."""
        tp = UserPrompt(prompt="What is the sum of 2 and 3?")
        assert tp.prompt == "What is the sum of 2 and 3?"

    def test_missing_prompt_key(self):
        """Vérifie qu'un objet sans la clé 'prompt' échoue."""
        with pytest.raises(ValidationError):
            UserPrompt(**{})

    def test_empty_prompt_string(self):
        """Vérifie qu'on refuse une chaîne vide pour le prompt."""
        with pytest.raises(ValidationError):
            UserPrompt(prompt="")


# =================================================================
# 3. OUTPUT TESTS (Validation de function_calling_results.json)
# =================================================================
class TestOutputValidation:

    def test_valid_output_format(self):
        """Vérifie le respect du format imposé par la consigne V.4.1."""
        data = {
            "prompt": "What is 2+2?",
            "name": "fn_add",
            "parameters": {"a": 2.0, "b": 2.0}
        }
        result = FunctionCallResult(**data)
        assert result.name == "fn_add"

    @pytest.mark.parametrize("missing_field", ["prompt", "name", "parameters"])
    def test_output_missing_fields(self, missing_field):
        """Vérifie que les 3 clés imposées par V.4.1 sont présentes."""
        data = {"prompt": "p", "name": "n", "parameters": {"a": 1}}
        del data[missing_field]
        with pytest.raises(ValidationError):
            FunctionCallResult(**data)

    def test_output_strict_types(self):
        """Vérifie que les types dans parameters sont préservés."""
        data = {
            "prompt": "p", "name": "n",
            "parameters": {
                "pi": 3.14,          # Float positif
                "count": 42,         # Int positif
                "temp": -10.5,       # Float négatif (le cas "signe moins")
                "balance": -500,     # Int négatif
                "label": "items",    # String
                "active": True       # Booléen
            }
        }

        result = FunctionCallResult(**data)
        
        assert result.parameters["pi"] == 3.14
        assert isinstance(result.parameters["pi"], float)
        
        assert result.parameters["temp"] == -10.5
        assert isinstance(result.parameters["temp"], float)
        
        assert result.parameters["balance"] == -500
        assert isinstance(result.parameters["balance"], int)
        
        assert isinstance(result.parameters["active"], bool)

    def test_output_invalid_type(self):
        """Vérifie que Pydantic refuse un type qui n'est pas dans l'Union (ex: une liste)."""
        data = {
            "prompt": "Test", 
            "name": "fn_add",
            "parameters": {"a": [1, 2, 3]} # Une liste n'est ni float, int, bool ou str
        }
        with pytest.raises(ValidationError):
            FunctionCallResult(**data)

    def test_output_forbid_extra_keys(self):
        """Vérifie qu'aucune clé superflue n'est acceptée (No extra keys or prose)."""
        data = {
            "prompt": "p", "name": "n", "parameters": {},
            "extra_info": "I am an AI"
        }
        with pytest.raises(ValidationError):
            FunctionCallResult(**data)