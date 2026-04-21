import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.constrained_decoder import ConstrainedDecoder

# On définit une constante pour les tests si elle n'est pas importée
NEG_INF = -1e10


@pytest.fixture
def decoder():
    """
    Fixture qui crée une instance de ConstrainedDecoder
    sans charger le LLM réel.
    """
    mock_llm = MagicMock()

    # On patche le __init__ pour éviter
    # les erreurs de "Repo ID" (chargement tokenizer/model)
    with patch.object(ConstrainedDecoder, '__init__', return_value=None):
        obj = ConstrainedDecoder(mock_llm)

        # On injecte manuellement les attributs nécessaires
        # au fonctionnement des méthodes
        obj.llm = mock_llm
        obj.quote_id = 99
        obj._tokens_num = {10, 11, 12}  # Ex: 10='1', 11='0', 12='.'
        obj._tokens_stop = {30, 31}    # Ex: 30=',', 31=' '

        return obj


def test_extract_param_value_number_formatting(decoder):
    """Vérifie que '10' est converti en 10.0 (float) avec le fix .0."""

    # Simulation de la séquence de tokens :
    # '1', '0', puis un token STOP (espace)
    side_effects = [10, 11, 31]
    decoder._get_masked_next = MagicMock(side_effect=side_effects)

    # Simulation du décodage de chaque token
    def mock_decode(token_ids):
        mapping = {10: "1", 11: "0", 31: " "}
        return mapping.get(token_ids[0], "")

    decoder.llm.decode = MagicMock(side_effect=mock_decode)

    # Exécution
    result = decoder.extract_param_value([1, 2, 3], "number")

    # Vérifications
    assert result == 10.0
    assert isinstance(result, float)
    # Vérifie que le décodeur a été appelé pour chaque token
    assert decoder.llm.decode.call_count == 2
    # '1' et '0' (le stop break avant le decode final)


def test_extract_param_value_string_newline_break(decoder):
    """
    Vérifie que la génération d'une string s'arrête en cas de saut de ligne.
    """

    # Simulation : l'IA commence à écrire puis fait un \n
    decoder.llm.get_logits_from_input_ids = MagicMock(
        return_value=np.zeros(100)
        )

    # On simule que le 2ème token généré est un saut de ligne
    tokens = [50, 51]  # 51 est l'ID de \n
    with patch("numpy.argmax", side_effect=tokens):
        def mock_decode(token_ids):
            return "abc" if token_ids == [50] else "\n"

        decoder.llm.decode = MagicMock(side_effect=mock_decode)

        result = decoder.extract_param_value([1], "string")

        # Le résultat doit contenir le premier token mais s'arrêter au \n
        assert result == "abc"


def test_string_first_token_masks_quote(decoder):
    """Vérifie que le guillemet est interdit au tout début d'une string."""

    # On crée un array NumPy rempli de zéros
    # Il est CRITIQUE que ce soit un array NumPy
    # pour que la modif [id] = NEG_INF fonctionne
    mock_logits = np.zeros(100, dtype=np.float32)

    # On mock la fonction pour qu'elle retourne notre array
    decoder.llm.get_logits_from_input_ids = MagicMock(return_value=mock_logits)

    # On simule un arrêt immédiat au second tour pour éviter la boucle infinie
    # Tour 0: on rend 50, Tour 1: on rend 99 (quote_id) pour stopper
    side_effects = [50, 99]

    with patch("numpy.argmax", side_effect=side_effects) as mock_argmax:
        # On mock aussi le decode pour éviter les messages
        # MagicMock dans la console
        decoder.llm.decode = MagicMock(return_value="a")

        decoder.extract_param_value([1], "string")

        # Le premier appel de argmax a reçu les logits modifiés
        # args[0] est l'array passé à np.argmax(logits)
        passed_logits = mock_argmax.call_args_list[0][0][0]

        # Assertion : on vérifie que l'index 99 est bien passé à NEG_INF
        assert passed_logits[99] <= -1e9


def test_number_with_existing_float_point(decoder):
    """Vérifie qu'on n'ajoute pas .0 si un point existe déjà."""

    # Séquence : '1', '.', '5', STOP
    side_effects = [10, 12, 13, 31]
    decoder._get_masked_next = MagicMock(side_effect=side_effects)

    def mock_decode(token_ids):
        mapping = {10: "1", 12: ".", 13: "5", 31: " "}
        return mapping.get(token_ids[0], "")

    decoder.llm.decode = MagicMock(side_effect=mock_decode)

    result = decoder.extract_param_value([1], "number")

    assert result == 1.5
    assert isinstance(result, float)


def test_empty_number_returns_zero(decoder):
    """Vérifie qu'une extraction de nombre vide renvoie 0.0."""

    # L'IA donne un stop token immédiatement
    decoder._get_masked_next = MagicMock(return_value=31)

    result = decoder.extract_param_value([1], "number")

    assert result == 0.0
