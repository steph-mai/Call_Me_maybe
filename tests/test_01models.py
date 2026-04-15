import pytest
from pydantic import ValidationError
from src.models import FunctionDefinition, UserPrompt, FunctionCallResult

# =================================================================
# 1. INPUT TESTS (Validation of function_definitions.json)
# =================================================================


class TestFunctionDefinition:
    """TestSuite for validating function definition structures."""

    def test_valid_function_definition(self) -> None:
        """Ensure a complete and correctly formatted definition is accepted."""
        data = {
            "name": "fn_add",
            "description": "Add two numbers",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "returns": {"type": "number"}
        }
        func = FunctionDefinition(**data)
        assert func.name == "fn_add"
        assert func.parameters["a"].type == "number"

    @pytest.mark.parametrize(
            "missing_field",
            ["name", "description", "parameters", "returns"]
             )
    def test_missing_fields(self, missing_field: str) -> None:
        """
        Verify that each mandatory field raises
        a ValidationError if missing.
        """
        base_data = {
            "name": "fn_add",
            "description": "Adds two numbers",
            "parameters": {"a": {"type": "number"}},
            "returns": {"type": "number"}
        }
        del base_data[missing_field]
        with pytest.raises(ValidationError):
            FunctionDefinition(**base_data)

    def test_completely_empty(self) -> None:
        """Ensure an empty dictionary triggers a ValidationError."""
        with pytest.raises(ValidationError):
            FunctionDefinition(**{})

    def test_no_parameters(self) -> None:
        """
        Validate that a function with an empty parameters object
        is allowed.
        """
        data = {
            "name": "fn_get_now",
            "description": "Get current time",
            "parameters": {},
            "returns": {"type": "string"}
        }
        func = FunctionDefinition(**data)
        assert func.parameters == {}

    def test_parameters_is_none(self) -> None:
        """Ensure that the 'parameters' field cannot be null."""
        data = {
            "name": "fn_test", "description": "d",
            "parameters": None, "returns": {"type": "s"}
        }
        with pytest.raises(ValidationError):
            FunctionDefinition(**data)

    def test_empty_name(self) -> None:
        """Verify that an empty string is rejected for the function name."""
        with pytest.raises(ValidationError):
            FunctionDefinition(
                name="",
                description="d",
                parameters={},
                returns={"type": "s"}
                )

    def test_invalid_return_format(self) -> None:
        """
        Ensure that 'returns' must be a structured object
        with a 'type' key.
        """
        with pytest.raises(ValidationError):
            FunctionDefinition(
                name="f",
                description="d",
                parameters={},
                returns="number")

    def test_parameter_type_unsupported(self) -> None:
        """
        Verify that only 'number', 'string', and 'boolean' types are accepted.
        """
        data = {
            "name": "fn_invalid", "description": "d",
            "parameters": {"arg1": {"type": "array"}},
            "returns": {"type": "s"}
        }
        with pytest.raises(ValidationError):
            FunctionDefinition(**data)

    def test_parameter_type_not_string(self) -> None:
        """Ensure that parameter types must be strings."""
        data = {
            "name": "fn_test", "description": "d",
            "parameters": {"arg1": {"type": 123}},
            "returns": {"type": "s"}
        }
        with pytest.raises(ValidationError):
            FunctionDefinition(**data)


# =================================================================
# 2. PROMPTS TESTS (Validation of function_calling_tests.json)
# =================================================================
class TestInputPrompts:
    """TestSuite for validating user prompt inputs."""

    def test_valid_prompt(self) -> None:
        """Verify that a standard user prompt is correctly validated."""
        tp = UserPrompt(prompt="What is the sum of 2 and 3?")
        assert tp.prompt == "What is the sum of 2 and 3?"

    def test_missing_prompt_key(self) -> None:
        """Ensure a ValidationError is raised if the prompt key is missing."""
        with pytest.raises(ValidationError):
            UserPrompt(**{})

    def test_empty_prompt_string(self) -> None:
        """Verify that an empty prompt string is rejected."""
        with pytest.raises(ValidationError):
            UserPrompt(prompt="")

    def test_none_prompt_string(self) -> None:
        """Ensure that the 'prompt' field cannot be null."""
        with pytest.raises(ValidationError):
            UserPrompt(prompt=None)

# =================================================================
# 3. OUTPUT TESTS (Validation of function_calling_results.json)
# =================================================================


class TestOutputValidation:
    """TestSuite for validating the final LLM output structure."""

    def test_valid_output_format(self) -> None:
        """Ensure the output follows the mandatory format specified."""
        data = {
            "prompt": "What is 2+2?",
            "name": "fn_add",
            "parameters": {"a": 2.0, "b": 2.0}
        }
        result = FunctionCallResult(**data)
        assert result.name == "fn_add"

    @pytest.mark.parametrize("missing_field", ["prompt", "name", "parameters"])
    def test_output_missing_fields(self, missing_field: str) -> None:
        """Verify that all three mandatory output keys must be present."""
        data = {"prompt": "p", "name": "n", "parameters": {"a": 1}}
        del data[missing_field]
        with pytest.raises(ValidationError):
            FunctionCallResult(**data)

    def test_output_strict_types(self) -> None:
        """Ensure that data types within parameters are strictly preserved."""
        data = {
            "prompt": "p", "name": "n",
            "parameters": {
                "pi": 3.14,
                "count": 42,
                "temp": -10.5,
                "balance": -500,
                "label": "items",
                "active": True
            }
        }
        result = FunctionCallResult(**data)

        assert isinstance(result.parameters["pi"], float)
        assert isinstance(result.parameters["count"], int)
        assert isinstance(result.parameters["temp"], float)
        assert isinstance(result.parameters["balance"], int)
        assert isinstance(result.parameters["active"], bool)

    def test_output_invalid_type(self) -> None:
        """Verify that unsupported types trigger a ValidationError."""
        data = {
            "prompt": "Test",
            "name": "fn_add",
            "parameters": {"a": [1, 2, 3]}
        }
        with pytest.raises(ValidationError):
            FunctionCallResult(**data)

    def test_output_forbid_extra_keys(self) -> None:
        """Ensure that no extra keys or prose are allowed in the output."""
        data = {
            "prompt": "p", "name": "n", "parameters": {},
            "extra_info": "I am an AI"
        }
        with pytest.raises(ValidationError):
            FunctionCallResult(**data)
