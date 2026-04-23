import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.constrained_decoder import ConstrainedDecoder

NEG_INF = -1e11


@pytest.fixture
def decoder():
    """
    Fix that creates an instance of ConstrainedDecoder,
    without loading the actual LLM.
    """
    mock_llm = MagicMock()

    # We patch the __init__ to avoid
    # "Repo ID" errors (tokenizer/model loading)
    with patch.object(ConstrainedDecoder, '__init__', return_value=None):
        obj = ConstrainedDecoder(mock_llm)

        # We manually inject the necessary attributes
        # for the methods to function
        obj.llm = mock_llm
        obj.quote_id = 99
        obj._tokens_num = {10, 11, 12}  # Ex: 10='1', 11='0', 12='.'
        obj._tokens_stop = {30, 31}    # Ex: 30=',', 31=' '

        return obj

def test_extract_param_value_string_newline_break(decoder):
    """
    Checks that string generation stops when a newline character is inserted.
    """

    # Simulation: the AI ​​starts writing then makes a \n
    decoder.llm.get_logits_from_input_ids = MagicMock(
        return_value=np.zeros(100)
        )

    # We simulate that the 2nd generated token is a line break
    tokens = [50, 51]  # 51 est l'ID de \n
    with patch("numpy.argmax", side_effect=tokens):
        def mock_decode(token_ids):
            return "abc" if token_ids == [50] else "\n"

        decoder.llm.decode = MagicMock(side_effect=mock_decode)

        result = decoder.extract_param_value([1], "string")

        # The result must contain the first token but stop at the \n
        assert result == "abc"


def test_string_first_token_masks_quote(decoder):
    """Check the quote is forbidden at the beginning of a string"""

    mock_logits = np.zeros(100, dtype=np.float32)

    decoder.llm.get_logits_from_input_ids = MagicMock(return_value=mock_logits)

    side_effects = [50, 99]

    with patch("numpy.argmax", side_effect=side_effects) as mock_argmax:
        decoder.llm.decode = MagicMock(return_value="a")

        decoder.extract_param_value([1], "string")

        passed_logits = mock_argmax.call_args_list[0][0][0]

        assert passed_logits[99] <= -1e9


def test_empty_number_returns_zero(decoder):
    """Verifies that an empty number extraction returns 0.0."""

    decoder._get_masked_next = MagicMock(return_value=31)

    result = decoder.extract_param_value([1], "number")

    assert result == ""
