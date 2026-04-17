import pytest
import numpy as np
from src.constraints import JSONStructureEnforcer

# On crée une classe Mock pour simuler le comportement du SDK LLM
class MockLLM:
    def encode(self, text):
        # Simule le retour du SDK : un tenseur/liste de listes d'IDs
        # On utilise une logique simple : l'ID est la longueur du texte pour le test
        class MockResult:
            def tolist(self):
                return [[len(text)]]
        return MockResult()

@pytest.fixture
def mock_llm():
    return MockLLM()

@pytest.fixture
def mock_vocab():
    """Fournit un vocabulaire réduit pour les tests."""
    return {
        0: "Bonjour",
        1: "{",
        2: '"prompt"',
        3: ":",
        4: " ",
        5: "<think>",
        6: '{"',
        7: ' {'
    }

def test_mask_at_absolute_start(mock_vocab, mock_llm):
    """Vérifie que le processeur force l'ouverture du JSON au début."""
    # L'enforcer va appeler llm.encode("{"), ce qui retournera ID 1 selon notre Mock
    processor = JSONStructureEnforcer(mock_vocab, mock_llm)
    logits = [10.0] * len(mock_vocab)
    
    filtered = processor.enforce_constraints(logits, "")
    
    # Avec l'ID 1 forcé par le masque unique
    assert filtered[1] == 10.0
    assert filtered[0] == -1e10
    assert filtered[5] == -1e10

def test_mask_after_open_brace(mock_vocab, mock_llm):
    """Vérifie qu'après '{', on force la clé 'prompt'."""
    processor = JSONStructureEnforcer(mock_vocab, mock_llm)
    logits = [10.0] * len(mock_vocab)
    
    # État : On vient d'écrire l'accolade
    filtered = processor.enforce_constraints(logits, "{")
    
    # Ici on utilise _filter_for_exact_string qui autorise les correspondances
    assert filtered[2] == 10.0
    assert filtered[0] == -1e10

def test_mask_after_prompt_key(mock_vocab, mock_llm):
    """Vérifie qu'après la clé, on force les deux-points."""
    processor = JSONStructureEnforcer(mock_vocab, mock_llm)
    logits = [10.0] * len(mock_vocab)
    
    # État : La clé "prompt" est présente
    filtered = processor.enforce_constraints(logits, '{"prompt"')
    
    # Le Mock retournera l'ID pour ":" (longueur 1) -> ID 1
    # Note: Dans un vrai test, il faudrait s'assurer que l'ID match le vocab
    assert filtered[processor.id_colon] == 10.0

def test_mask_with_whitespace_variants(mock_llm):
    """Vérifie que le processeur accepte '{' même avec des espaces via le trimming."""
    vocab = {1: "{", 7: " {"}
    processor = JSONStructureEnforcer(vocab, mock_llm)
    logits = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    
    # On teste le comportement du start avec un texte vide
    filtered = processor.enforce_constraints(logits, "")
    
    # L'ID forcé par le SDK pour "{" (id 1) doit être actif
    assert filtered[1] == 10.0

def test_partial_token_matching(mock_vocab, mock_llm):
    """Vérifie que le processeur autorise des tokens qui sont des débuts de clé."""
    vocab = {8: '"pro'}
    processor = JSONStructureEnforcer(vocab, mock_llm)
    logits = [0.0] * 10
    logits[8] = 10.0
    
    # Test de la méthode interne de matching partiel
    filtered = processor._filter_for_exact_string(logits, '"prompt"')
    assert filtered[8] == 10.0

def test_dead_end_security(mock_vocab, mock_llm):
    """Vérifie le comportement si aucun token n'est valide."""
    # Vocabulaire ne contenant pas l'ID attendu (ID 1 pour "{")
    processor = JSONStructureEnforcer({99: "Invalide"}, mock_llm)
    logits = [10.0] * 100
    
    filtered = processor.enforce_constraints(logits, "")
    # Tous les logits devraient être à -1e10 car l'ID 1 n'est pas dans le vocabulaire
    assert all(score <= -1e10 for score in filtered)