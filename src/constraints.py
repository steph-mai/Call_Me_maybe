from typing import List

class JSONStructureEnforcer:
    def __init__(self, vocab: dict, control_tokens: dict, functions: list):
        self.vocab = vocab
        self.tokens = control_tokens
        self.functions = functions
        self.mask_value = -1e10

    def enforce_constraints(self, logits: List[float], current_text: str, mode: str, **kwargs) -> List[float]:
        if mode == "name_only":
            return self._apply_mask(logits, ["function_names"])
            
        if mode == "params_only":
            # Sécurité maximale pour récupérer les clés
            fn = kwargs.get("current_fn")
            keys = []
            if isinstance(fn, dict):
                # On utilise .get() pour ne jamais lever de KeyError
                params_obj = fn.get("parameters", {})
                if isinstance(params_obj, dict):
                    keys = list(params_obj.keys())

            # Si on attend une clé après l'ouverture '{' ou une virgule ', '
            if current_text.endswith('{') or current_text.endswith(', '):
                # On autorise les tokens de type "quotes" pour commencer une clé
                return self._apply_mask(logits, ["quotes"])
                
            # Si on attend une valeur après ': '
            if current_text.endswith(': '):
                # Si c'est fn_add_numbers on force les chiffres, sinon quotes
                if fn and "add" in str(fn.get("name", "")):
                    return self._apply_mask(logits, ["digits"])
                return self._apply_mask(logits, ["quotes"])

            # Par défaut pour les paramètres on autorise les types de base du JSON
            return self._apply_mask(logits, ["digits", "quotes", "comma", "brace_close"])

        return logits

    def _apply_mask(self, logits, keys_list):
        new_logits = [self.mask_value] * len(logits)
        # On ajoute toujours "whitespace" pour que le modèle puisse respirer
        for k in keys_list + ["whitespace"]:
            for tid in self.tokens.get(k, []):
                if 0 <= tid < len(new_logits):
                    new_logits[tid] = logits[tid]
        return new_logits