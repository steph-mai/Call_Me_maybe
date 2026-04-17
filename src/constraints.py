from typing import List

class JSONStructureEnforcer:
    def __init__(self, vocab: dict, llm):
        self.vocab = vocab
        self.llm = llm
        self.mask_value = -1e10
        
        # Pré-encodage via le SDK
        self.id_colon = self._get_id(":")
        self.id_brace_open = self._get_id("{")
        self.id_quote = self._get_id('"')
        self.id_colon_space_brace = self._get_id(": {")

    def _get_id(self, text: str) -> int:
        """Récupère l'ID du token via le SDK."""
        return self.llm.encode(text).tolist()[0][-1]

    def _mask_single_token(self, logits: List[float], target_id: int) -> List[float]:
        """Force un seul token."""
        mask = [self.mask_value] * len(logits)
        if 0 <= target_id < len(logits):
            mask[target_id] = logits[target_id]
        return mask

    def enforce_constraints(self, logits: List[float], current_text: str) -> List[float]:
        text = current_text
        trimmed = text.strip()

        # 1. DÉMARRAGE : On doit utiliser le filtre de texte pour autoriser '{' ET '{"'
        if not trimmed:
            return self._filter_for_exact_string(logits, "{")
        
        # 2. STRUCTURE GLOBALE
        if trimmed == "{":
            return self._filter_for_exact_string(logits, '"prompt"')

        if text.endswith('"prompt"'):
            return self._mask_single_token(logits, self.id_colon)
        
        if text.endswith('"prompt":') or text.endswith('"prompt": '):
            return self._mask_single_token(logits, self.id_quote)

        # Transition "name"
        if text.count('"') == 4 and text.endswith('"') and '"name"' not in text:
            return self._filter_for_exact_string(logits, ', "name"')

        if text.endswith('"name"'):
            return self._mask_single_token(logits, self.id_colon)

        # Transition "parameters"
        if text.count('"') == 6 and text.endswith('"') and '"parameters"' not in text:
            return self._filter_for_exact_string(logits, ', "parameters"')

        if text.endswith('"parameters"'):
            return self._mask_single_token(logits, self.id_colon)
        
        if text.endswith('"parameters":') or text.endswith('"parameters": '):
            return self._mask_single_token(logits, self.id_brace_open)

        # 3. GESTION DES PARAMÈTRES
        if '"parameters":' in text:
            is_math_fn = any(fn in text for fn in ["fn_add_numbers", "fn_get_square_root"])
            
            if text.endswith('":') or text.endswith('": '):
                if is_math_fn:
                    return self._filter_for_numbers_only(logits)
                else:
                    return self._mask_single_token(logits, self.id_quote)

            if is_math_fn and text and text[-1].isdigit():
                return self._filter_for_numbers_or_separator(logits)
            
            if is_math_fn and text.endswith(','):
                return self._mask_single_token(logits, self.id_quote)

        return logits

    def _filter_for_numbers_only(self, logits: List[float]) -> List[float]:
        new_logits = [self.mask_value] * len(logits)
        for tid, t_content in self.vocab.items():
            t_str = str(t_content).strip()
            if t_str and t_str.isdigit() and '"' not in t_str:
                new_logits[tid] = logits[tid]
        return new_logits

    def _filter_for_numbers_or_separator(self, logits: List[float]) -> List[float]:
        new_logits = [self.mask_value] * len(logits)
        allowed = set("0123456789, }")
        for tid, t_content in self.vocab.items():
            t_str = str(t_content)
            if t_str and all(c in allowed for c in t_str) and '"' not in t_str:
                new_logits[tid] = logits[tid]
        return new_logits

    def _filter_for_exact_string(self, logits: List[float], target: str) -> List[float]:
        """Méthode compatible avec les tests Pytest (Partial Matching)."""
        new_logits = [self.mask_value] * len(logits)
        for tid, t_content in self.vocab.items():
            t_str = str(t_content).strip()
            if not t_str: continue
            # On autorise si le token est une partie de la cible ou contient la cible
            if target.startswith(t_str) or t_str.startswith(target):
                new_logits[tid] = logits[tid]
        return new_logits
    
    # Alias pour le test qui cherche spécifiquement ce nom
    _mask_for_exact_string = _filter_for_exact_string
# from typing import List


# class JSONStructureEnforcer:
#     """
#     Enforces strict JSON formatting on LLM outputs by applying 
#     grammar-based constraints to prediction logits.
#     """
#     def __init__(self, vocab: dict):
#         self.vocab = vocab
#         self.token_to_id = {str(v): k for k, v in vocab.items()}
  
#     def enforce_constraints(self,
#                             logits: List[float],
#                             current_text: str
#                             ) -> List[float]:
#         """
#         Constrain logits to ensure the next token conforms
#         to valid JSON syntax.

#         This method implements a state machine to filter the LLM output
#         based on the current generation context, allowing only tokens
#         that maintain the required structural integrity.

#         Args:
#             logits: A list of raw probability scores from the LLM.
#             current_text: The string generated by the model so far.

#         Returns:
#             The filtered list of logits with invalid token scores
#             set to -infinity.
#         """
#         text = current_text.strip()

#         if not text:
#             return self._filter_for_start(logits)

#         if text.endswith('{') and '"prompt"' not in text:
#             return self._filter_for_exact_string(logits, '"prompt"')

#         if text.endswith('"prompt"'):
#             return self._filter_for_exact_string(logits, ":")

#         if text.endswith('"prompt:"'):
#             return self._filter_for_exact_string(logits, '"')

#         if text.count('"') == 2 and text.endswith('"') and '"name"' not in text:
#             return self._filter_for_exact_string(logits, '"name"')

#         return logits


#     def _filter_for_start(self, logits: List[float]) -> List[float]:
#         """
#         Apply a mask that only allows tokens starting with '{'.

#         This prevents the model from starting its response with
#         conversational fillers or thinking tags.

#         Args:
#             logits: Current logit distribution from the LLM.

#         Returns:
#             Logits with a massive penalty applied to all non-compliant tokens.
#         """
#         new_logits = [-1e10] * len(logits)
#         found = False
#         for token_id, t_content in self.vocab.items():
#             #PQ transformé en str
#             t_str = str(t_content).strip()
#             if t_str == '{':
#                 new_logits[token_id] = logits[token_id]
#                 found = True
#         return new_logits if found else logits

#     def _filter_for_exact_string(
#             self,
#             logits: List[float],
#             target: str
#             ):
#         """Authorize only logits that lead to the target."""
#         new_logits = [-1e10] * len(logits)
#         found = False

#         for token_id, t_content in self.vocab.items():
#             token_str = str(t_content).strip()
#             if not token_str: continue
            
#             # Si on attend "prompt", on refuse "EIF" ou "Okay"
#             # On vérifie si le token est une sous-partie de notre cible
#             if target.startswith(token_str) or token_str.startswith(target):
#                 new_logits[token_id] = logits[token_id]
#                 found = True

#         return new_logits if found else logits