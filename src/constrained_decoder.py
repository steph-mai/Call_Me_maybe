import json
import re
import sys
import numpy as np
from typing import Any, List, Set
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from src.state_node import StateNode

NEG_INF: float = -1e11


class ConstrainedDecoder:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.llm = Small_LLM_Model(model_name=model_name)

        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self._tokens_num = self._build_set(r'^[0-9.\-eE]+$')
        self._tokens_stop = self._build_set(r'^[,\}\]:\s\n\t]+$')
        self.quote_id = self.llm.encode('"')[0].tolist()[-1]

    def _build_set(self, pattern: str) -> Set[int]:
        allowed = set()
        for tok_str, tid in self.vocab.items():
            clean = tok_str.replace("\u0120", "").replace(" ", "").strip()
            if clean and re.match(pattern, clean):
                allowed.add(tid)
        return allowed

    def _get_masked_next(
            self,
            ids: List[int],
            allowed: Set[int] | None
            ) -> int:
        # Conversion : On transforme cela en tableau NumPy pour faire
        # des calculs rapides. float32 : definit le type de donnees
        # et le nombre de bits utilises pour les stocker
        logits = np.array(
            self.llm.get_logits_from_input_ids(ids),
            dtype=np.float32
        )
        # Le dernier mot : Le modèle renvoie souvent les scores pour toute la
        # phrase. on ne veut que les prédictions pour le tout dernier token
        # (celui qui va être généré). On réduit donc le tableau pour ne garder
        # Le shape precise la forme du modele renvoyé.
        while len(logits.shape) > 1:
            logits = logits[-1]
        # on remplit un tableau de même forme que celui des logits
        # avec des NEG_INF
        mask = np.full(logits.shape, NEG_INF, dtype=np.float32)
        # len(mask) représente la taille totale du dictionnaire du modèle
        # (par exemple, 32 000 tokens).
        # La condition if tid < len(mask): vérifie que l'identifiant
        # du token (tid) que l on veut autoriser existe bien
        # dans l'index du modèle.
        for tid in allowed:
            if tid < len(mask):
                mask[tid] = logits[tid]

        return int(np.argmax(mask))

    def force_name(self,
                   full_prompt_ids: List[int],
                   functions: List[FunctionDefinition]
                   ) -> str:
        root = StateNode()
        for f in functions:
            root.insert_name(self.llm.encode(f.name)[0].tolist(), f.name)
        curr = root
        current_context_ids = list(full_prompt_ids)
        while True:
            allowed = set(curr.children.keys())
            # tant qu il y a des enfants
            if not allowed:
                break
            # Le Masquage: S'il y a plusieurs choix possibles (ex: deux
            # fonctions commencent par "fn_"),
            # on appelle _get_masked_next.
            # Le modèle d'IA va choisir le token le plus probable uniquement
            # parmi ceux autorisés.
            # S'il n'y a qu'un seul choix,
            # on ne demande même pas à l'IA (gain de temps), on prend
            # le seul token dispo (next(iter(allowed))).
            if len(allowed) > 1:
                chosen = self._get_masked_next(current_context_ids, allowed)
            else:
                # iter transforme le set en iterable
                # avec next on recupere le 1er(ici le seul élément)
                chosen = next(iter(allowed))

            current_context_ids.append(chosen)

            curr = curr.children[chosen]

            if curr.is_terminal and not curr.children:
                return curr.name
        return curr.name

    def extract_param_value(self, ids: List[int], p_type: str) -> Any:
        current_context_ids = list(ids)
        extracted_value = ""

        if p_type == "number":
            allowed = self._tokens_num | self._tokens_stop
            max_tokens = 20
        else:
            allowed = None
            max_tokens = 200

        for i in range(max_tokens):
            if p_type == "number":
                chosen = self._get_masked_next(current_context_ids, allowed)
            else:
                logits = np.array(
                    self.llm.get_logits_from_input_ids(current_context_ids),
                    dtype=np.float32
                )
                # tant qu il reste pls dimensions à la structure,
                # on choisit son dernier element
                while len(logits.shape) > 1:
                    logits = logits[-1]
                if i == 0:
                    logits[self.quote_id] = NEG_INF  # Empêche de fermer direct
                chosen = int(np.argmax(logits))

            # Pour les str, comme on a forcé l'ouverture d'un guillemet
            # au début de l'instruction ("),
            # l'IA doit normalement fermer ce guillemet
            # pour signaler la fin de sa réponse.
            # Pour les numbers : (p_type == "number" and
            # chosen in self._tokens_stop) : L'IA s'arrête lorsqu'elle tape
            # un caractère qui ne fait plus partie du nombre
            # (un espace, une virgule, ou un saut de ligne).
            # Si on attend un nombre ET que le token est dans ta liste
            # de "caractères d'arrêt", on valide la fin de la saisie.
            is_stop_token = (
                (chosen == self.quote_id) or
                (p_type == "number" and chosen in self._tokens_stop)
            )
            if is_stop_token:
                if p_type == "number" and extracted_value:
                    if (
                        "." not in extracted_value
                        and "e" not in extracted_value.lower()
                    ):
                        extracted_value += ".0"
                break

            decoded_token = self.llm.decode([chosen])

            # Sécurité "anti_deraillage" pour le LLM
            if "\n" in decoded_token and i > 0:
                break

            extracted_value += decoded_token
            current_context_ids.append(chosen)

        if p_type == "number":
            try:
                return float(extracted_value) if extracted_value else 0.0
            except ValueError:
                return 0.0

        return extracted_value.strip()
