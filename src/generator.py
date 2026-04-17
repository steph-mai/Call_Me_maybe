import json
import re
from src.prompt_builder import PromptBuilder

class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        from llm_sdk import Small_LLM_Model
        self.llm = Small_LLM_Model(model_name=model_name)
        
        # Chargement vocabulaire (ID: Token) - Ton bloc
        v_path = self.llm.get_path_to_vocab_file()
        with open(v_path, mode='r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
        self.vocab = self._normalize_vocab(raw_vocab)

        self.prompt_builder = PromptBuilder()

    def _normalize_vocab(self, raw_vocab):
        vocab = {}
        for k, v in raw_vocab.items():
            try: vocab[int(k)] = str(v)
            except ValueError: vocab[int(v)] = str(k)
        return vocab

    def _get_control_tokens(self, mode: str, whitelist: list = None):
        """On adapte ta logique pour filtrer par TYPE."""
        tokens = {"target": [], "whitespace": []}
        
        for tid, t_text in self.vocab.items():
            # Gestion des espaces pour la fluidité
            if not t_text.strip() and len(t_text) > 0:
                tokens["whitespace"].append(tid)
                continue

            clean = t_text.strip().replace('"', '').replace("'", "")
            
            if mode == "whitelist" and whitelist:
                if any(clean in fn for fn in whitelist):
                    tokens["target"].append(tid)
            elif mode == "number":
                if any(char.isdigit() or char == "." for char in clean):
                    tokens["target"].append(tid)
            else: # mode "text" ou "string"
                tokens["target"].append(tid) # Plus permissif pour les chaînes
                
        return tokens

    def _run_constrained_generation(self, prompt: str, mode: str, whitelist: list = None):
        """Ta boucle de génération sécurisée."""
        prompt_ids = self.llm.encode(prompt).tolist()[0]
        generated_text = ""
        tokens = []
        controls = self._get_control_tokens(mode, whitelist)
        eos_id = getattr(self.llm._tokenizer, 'eos_token_id', -1)

        # On limite à 15 tokens pour une valeur unique (évite le bégaiement)
        for _ in range(15):
            raw_logits = self.llm.get_logits_from_input_ids(prompt_ids + tokens)
            
            # Ton FIX navigation listes Python
            logits_vec = raw_logits
            while isinstance(logits_vec, list) and len(logits_vec) > 0 and isinstance(logits_vec[0], list):
                logits_vec = logits_vec[-1]

            mask_value = -1e11
            new_logits = [mask_value] * len(logits_vec)
            
            # Autorisation : Valeurs cibles + Espaces + EOS
            allowed_ids = set(controls["target"] + controls["whitespace"])
            if eos_id != -1: allowed_ids.add(eos_id)

            for tid in allowed_ids:
                if 0 <= tid < len(new_logits):
                    new_logits[tid] = logits_vec[tid]

            max_val = max(new_logits)
            if max_val <= mask_value: break
                
            next_id = new_logits.index(max_val)
            if next_id == eos_id: break
            
            t_text = self.llm.decode([next_id])
            
            # Stop si espace après le texte (fin de la valeur)
            if generated_text.strip() and (not t_text.strip() or "\n" in t_text):
                break
                
            tokens.append(next_id)
            generated_text += t_text
            print(t_text, end="", flush=True)

        return generated_text.strip()

    def generate(self, query: str, functions: list) -> dict:
        """La Machine à États : On pilote l'extraction pas à pas."""
        
        # --- ÉTAPE 1 : Sélection du Nom ---
        name_prompt = self.prompt_builder.build_name_selector_prompt(query, functions)
        fn_names = [f["name"] for f in functions]
        raw_name = self._run_constrained_generation(name_prompt, "whitelist", whitelist=fn_names)
        
        # On valide le nom par rapport à ta liste
        selected_name = next((n for n in fn_names if n in raw_name), fn_names[0])

        # --- ÉTAPE 2 : Remplissage des Paramètres (FSM) ---
        fn_info = next((f for f in functions if f["name"] == selected_name), None)
        extracted_params = {}

        if fn_info:
            params_config = fn_info.get("parameters", {})
            
            # On ne laisse pas le LLM écrire le JSON. 
            # On boucle sur TON dictionnaire pour lui poser des questions ciblées.
            for p_name, p_info in params_config.items():
                p_type = p_info.get("type", "string")
                
                # On construit un mini-prompt pour CHAQUE clé
                mini_prompt = (
                    f"### QUERY: {query}\n"
                    f"### TASK: Extract the value for '{p_name}' (Type: {p_type})\n"
                    f"### RESPONSE\n{p_name}: "
                )
                
                # On force le mode (number ou text) basé sur TON dico
                mode_key = "number" if p_type == "number" else "text"
                val_raw = self._run_constrained_generation(mini_prompt, mode_key)
                
                # Conversion propre
                if p_type == "number":
                    nums = re.findall(r'\d+\.?\d*', val_raw)
                    extracted_params[p_name] = float(nums[0]) if nums else 0
                else:
                    extracted_params[p_name] = val_raw.strip('"').strip()

        # Retourne le dictionnaire Python final (Zéro parsing JSON nécessaire)
        return {
            "prompt": query,
            "name": selected_name,
            "parameters": extracted_params
        }