import pytest
from src.constraints import JSONStructureEnforcer

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
        6: '{"'
    }

def test_mask_at_absolute_start(mock_vocab):
    """Vérifie que le processeur force l'ouverture du JSON au début."""
    processor = JSONStructureEnforcer(mock_vocab)
    logits = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0] # Tous les scores sont égaux
    
    # Au début (texte vide)
    filtered = processor.enforce_constraints(logits, "")
    
    # Seuls les tokens commençant par '{' doivent être conservés
    assert filtered[1] == 10.0  # "{"
    assert filtered[6] == 10.0  # '{"'
    assert filtered[0] <= -1e10  # "Bonjour" masqué
    assert filtered[5] <= -1e10  # "<think>" masqué

def test_mask_after_open_brace(mock_vocab):
    """Vérifie qu'après '{', on force la clé 'prompt'."""
    processor = JSONStructureEnforcer(mock_vocab)
    logits = [10.0] * len(mock_vocab)
    
    # État : On vient d'écrire l'accolade
    filtered = processor.enforce_constraints(logits, "{")
    
    # Seul '"prompt"' (id 2) devrait être autorisé
    assert filtered[2] == 10.0
    assert filtered[0] <= -1e10
    assert filtered[1] <= -1e10

def test_mask_after_prompt_key(mock_vocab):
    """Vérifie qu'après la clé, on force les deux-points."""
    processor = JSONStructureEnforcer(mock_vocab)
    logits = [10.0] * len(mock_vocab)
    
    # État : La clé "prompt" est présente
    filtered = processor.enforce_constraints(logits, '{"prompt"')
    
    # Seul ":" (id 3) devrait être autorisé
    assert filtered[3] == 10.0
    assert filtered[2] <= -1e10

def test_mask_with_whitespace_variants(mock_vocab):
    """Vérifie que le processeur accepte '{' même avec des espaces."""
    # On ajoute un token avec espace dans un mock_vocab étendu
    extended_vocab = {1: "{", 7: " {"} 
    processor = JSONStructureEnforcer(extended_vocab)
    logits = [10.0] * 8
    
    filtered = processor.enforce_constraints(logits, "")
    
    # Les deux doivent être autorisés
    assert filtered[1] == 10.0
    assert filtered[7] == 10.0

def test_partial_token_matching(mock_vocab):
    """Vérifie que le processeur autorise des tokens qui sont des débuts de clé."""
    # Imagine que le vocabulaire a un token '"pro'
    vocab = {8: '"pro'}
    processor = JSONStructureEnforcer(vocab)
    logits = [10.0] * 9
    
    # Si on attend '"prompt"', le token '"pro' doit être autorisé
    filtered = processor._filter_for_exact_string(logits, '"prompt"')
    assert filtered[8] == 10.0

def test_dead_end_security(mock_vocab):
    """Vérifie le comportement si aucun token n'est valide."""
    processor = JSONStructureEnforcer({0: "Invalide"})
    logits = [10.0]
    
    filtered = processor.enforce_constraints(logits, "")
    # Tous les logits devraient être à -1e10
    assert all(score <= -1e10 for score in filtered)
