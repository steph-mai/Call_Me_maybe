import json
import re
import sys
import numpy as np
from typing import Any, Dict, List, Set
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, UserPrompt, FunctionCallResult

NEG_INF: float = -1e11


class TrieNode:
    def __init__(self):
        self.children: Dict[int, 'TrieNode'] = {}
        self.is_terminal: bool = False
        self.name: str = ""

    def insert(self, token_ids: List[int], name: str):
        node = self
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_terminal = True
        node.name = name


class ConstrainedDecoder:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.llm = Small_LLM_Model(model_name=model_name)
        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
            
        self._tokens_num = self._build_set(r'^[0-9.\-eE]+$')
        self._tokens_stop = self._build_set(r'^[,\}\]:\s\n\t]+$')
        self._quote_id = self.llm.encode('"')[0].tolist()[-1]

    def _build_set(self, pattern: str) -> Set[int]:
        allowed = set()
        for tok_str, tid in self.vocab.items():
            clean = tok_str.replace("\u0120", "").replace(" ", "").strip()
            if clean and re.match(pattern, clean):
                allowed.add(tid)
        return allowed

    def _get_masked_next(self, ids: List[int], allowed: Set[int]) -> int:
        logits = np.array(self.llm.get_logits_from_input_ids(ids), dtype=np.float32)
        while len(logits.shape) > 1: logits = logits[-1]
        mask = np.full(logits.shape, NEG_INF, dtype=np.float32)
        for tid in allowed:
            if tid < len(mask): mask[tid] = logits[tid]
        return int(np.argmax(mask))

    def _force_name(self, ids: List[int], functions: List[FunctionDefinition]) -> str:
        root = TrieNode()
        for f in functions:
            root.insert(self.llm.encode(f.name)[0].tolist(), f.name)
        curr = root
        temp_ids = list(ids)
        sys.stdout.write("\n  [NAME]: ")
        while True:
            allowed = set(curr.children.keys())
            if not allowed: break
            chosen = self._get_masked_next(temp_ids, allowed) if len(allowed) > 1 else next(iter(allowed))
            sys.stdout.write(f"\033[94m{self.llm.decode([chosen])}\033[0m")
            sys.stdout.flush()
            temp_ids.append(chosen)
            curr = curr.children[chosen]
            if curr.is_terminal and not curr.children: return curr.name
        return curr.name

    def _extract_value(self, ids: List[int], p_type: str) -> Any:
        temp_ids = list(ids)
        res = ""

        if p_type == "number":
            allowed = self._tokens_num | self._tokens_stop
            for _ in range(15):
                tid = self._get_masked_next(temp_ids, allowed)
                
                # Si le LLM choisit un token d'arrêt (espace, virgule, etc.)
                if tid in self._tokens_stop:
                    break
                
                chunk = self.llm.decode([tid])
                clean = "".join(c for c in chunk if c in "0123456789.eE-")
                
                if not clean: # Si le token décodé n'est pas numérique
                    break
                
                sys.stdout.write(f"\033[92m{chunk}\033[0m")
                sys.stdout.flush()
                res += chunk
                temp_ids.append(tid)
            
            # --- FIX : FORCER LE FORMAT .0 ---
            if res:
                # Si c'est un entier (pas de point, pas de 'e')
                if "." not in res and "e" not in res.lower():
                    sys.stdout.write(f"\033[92m.0\033[0m")
                    res += ".0"
                return float(res)
            return 0.0
        
        else: # STRINGS (Fix pour les Regex incomplets)
            for i in range(120):
                logits = np.array(self.llm.get_logits_from_input_ids(temp_ids), dtype=np.float32)
                while len(logits.shape) > 1: logits = logits[-1]
                
                if i == 0:
                    logits[self._quote_id] = NEG_INF
                    for tid in self._tokens_stop:
                        if tid < len(logits): logits[tid] = NEG_INF
                
                chosen = int(np.argmax(logits))
                if chosen == self._quote_id: break
                
                chunk = self.llm.decode([chosen])
                # On ne break sur \n que si on a déjà du texte (évite les sorties prématurées)
                if "\n" in chunk and len(res) > 0: break
                
                sys.stdout.write(f"\033[93m{chunk}\033[0m")
                sys.stdout.flush()
                res += chunk
                temp_ids.append(chosen)
                
            return res.strip()

    def process_prompt(self, prompt: str, functions: List[FunctionDefinition], static_ids: List[int]) -> FunctionCallResult:
        # 1. Sélection du Nom
        name_ids = static_ids + self.llm.encode(f"\nQuery: {prompt}\nFunction:")[0].tolist()
        selected_name = self._force_name(name_ids, functions)
        fn_def = next(f for f in functions if f.name == selected_name)

        # 2. Extraction des Paramètres (Context Reset)
        final_params = {}
        for p_name, p_info in fn_def.parameters.items():
            sys.stdout.write(f" | {p_name}: ")

            # Utilisation de l'ancre JSON pour stabiliser l'extraction
            instruction = f"\nTask: {prompt}\nFunction: {selected_name}\n{p_name}="
            if p_info.type == "string": instruction += '"'

            p_ids = static_ids + self.llm.encode(instruction)[0].tolist()
            val = self._extract_value(p_ids, p_info.type)

            if isinstance(val, str) and p_name == "regex":
                # On ferme les parenthèses manquantes
                while val.count('(') > val.count(')'):
                    val += ')'
                # On ferme les crochets manquants
                while val.count('[') > val.count(']'):
                    val += ']'

                # Nettoyage de sécurité : si le modèle a inclus un guillemet de fin par erreur
                val = val.replace('"', '').strip()

            if p_name == "replacement" and isinstance(val, str):
                # Si l'utilisateur demande des asterisks et que le modèle en met trop
                # On réduit les suites d'étoiles à une seule étoile
                if "**" in val:
                    val = "*"
                val = val.replace('"', '').strip()

            final_params[p_name] = val

        return FunctionCallResult(prompt=prompt, name=selected_name, parameters=final_params)

    def run(self, functions: List[FunctionDefinition], callables: List[UserPrompt], output_path: str):
        # FIX : Utilisation de model_dump() pour la sérialisation JSON
        tools_list = [f.model_dump() for f in functions]
        static_ids = self.llm.encode(f"System: Tool Extractor. Tools: {json.dumps(tools_list)}")[0].tolist()

        results = []
        for idx, call in enumerate(callables):
            sys.stdout.write(f"\n[{idx+1}/{len(callables)}] Prompt: {call.prompt[:30]}...")
            res = self.process_prompt(call.prompt, functions, static_ids)
            results.append(res.model_dump())
            sys.stdout.flush()
            
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)