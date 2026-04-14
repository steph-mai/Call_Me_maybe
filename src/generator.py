from typing import List
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionCallResult, UserPrompt
import json


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.llm = Small_LLM_Model(model_name=model_name)
        voca_path = self.llm.get_path_to_vocab_file()

        with open(voca_path, mode='r', encoding='utf-8') as f:
            raw_vocab = json.load(f)

        # On crée un dictionnaire propre : { ID_ENTIER: "TEXTE_TOKEN" }
        self.vocab = {}
        for key, value in raw_vocab.items():
            # On essaie de voir si la clé est l'ID (Cas 1) ou si la valeur est l'ID (Cas 2)
            try:
                # Si key est l'ID (ex: "123": "token")
                v_id = int(key)
                v_token = str(value)
            except ValueError:
                # Si c'est la valeur qui est l'ID (ex: "token": 123)
                v_id = int(value)
                v_token = str(key)
            
            self.vocab[v_id] = v_token

    def _build_full_prompt(self,
                           prompt_data: UserPrompt,
                           functions: List[FunctionDefinition]
                           ) -> str:
        """
        Constructs the complete prompt
        with system instructions, tools, and user query.
        """
        tools_desc = ""
        for f in functions:
            tools_desc += f"- Name: {f.name}\n"
            tools_desc += f"  Description: {f.description}\n"
            tools_desc += f"  Parameters: {f.parameters}\n\n"

        system_prompt = (
            "### INSTRUCTIONS\n"
            "You are a specialized function-calling engine. Your ONLY task is to map user queries to specific JSON tool calls.\n"
            "STRICT RULE: You must output valid JSON and NOTHING ELSE. No thinking, no intro, no outro.\n\n"
            
            "### AVAILABLE FUNCTIONS\n"
            "You must choose one function from this list based on the user's need:\n"
            f"{tools_desc}\n"
            
            "### OUTPUT FORMAT\n"
            "Your response must follow this EXACT schema:\n"
            "{\n"
            '  "prompt": "The exact user query",\n'
            '  "name": "function_name",\n'
            '  "parameters": {"param_name": value}\n'
            "}\n\n"
            
            "### CRITICAL CONSTRAINTS\n"
            "- Never include <think> tags.\n"
            "- Never explain your choice.\n"
            "- If no function matches, use the most relevant one or a generic fallback if provided."
        )

        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_data.prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return full_prompt

    def generate(self, prompt_data: UserPrompt, functions: List[FunctionDefinition]) -> FunctionCallResult:
        prompt_str = self._build_full_prompt(prompt_data, functions)
        # tokenisation du prompt > 
        # on transforme le tenseur renvoye par la fonction encode()
        # en liste (initialement de type[[]]
        # on prend son premier élément = le premier prompt avec [0]
        input_ids = self.llm.encode(prompt_str).tolist()[0]

        generated_tokens = []
        for _ in range(100): # Sécurité : max 200 tokens pour éviter l'infini
            current_sequence = input_ids + generated_tokens
            
            logits = self.llm.get_logits_from_input_ids(current_sequence)
            
            current_text = self.llm.decode(generated_tokens)
            logits = self._apply_constraints(logits, current_text, functions)
            next_token_id = logits.index(max(logits))
            if next_token_id == self.llm._tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id)

        raw_json_str = self.llm.decode(generated_tokens)

        return self._parse_and_validate(raw_json_str, prompt_data.prompt)




    def _apply_constraints(self, logits: List[float], current_text: str, functions: List[FunctionDefinition]) -> List[float]:
        def _apply_constraints(self, logits: List[float], current_text: str, functions: List[FunctionDefinition]) -> List[float]:
        if not current_text.strip():
            new_logits = [-1e10] * len(logits)

            for token_id, token_content in self.vocab.items():
                # On vérifie si l'ID est bien dans la plage des logits (sécurité)
                if token_id < len(logits):
                    if token_content.strip().startswith('{'):
                        new_logits[token_id] = logits[token_id]

            return new_logits

        return logits

    
    def _clean_output(raw_output: str) -> str:
        pass
# Rôle : Nettoyage de secours.
# Détail : Supprimer les espaces inutiles ou les caractères de fin de chaîne que le LLM aurait pu ajouter malgré la contrainte.

    def _parse_and_validate(self, json_str: str, original_prompt: str) -> FunctionCallResult:
        # DEBUG : Affiche ce que l'IA a vraiment écrit
        print(f"\n--- DEBUG RAW OUTPUT ---\n'{json_str}'\n-----------------------")
        
        if not json_str.strip():
            raise ValueError("L'IA n'a rien généré du tout (chaîne vide).")

        try:
            # On essaie de trouver le JSON si l'IA a mis du texte avant/après
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = json_str[start:end]

            data = json.loads(json_str)
            data["prompt"] = original_prompt
            return FunctionCallResult(**data)
        except Exception as e:
            # On affiche l'erreur ET le contenu problématique
            raise ValueError(f"Echec du parsing. Contenu reçu: {json_str} | Erreur: {e}")