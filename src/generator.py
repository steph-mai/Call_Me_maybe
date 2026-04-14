# genere le format JSON attendu ({
# "prompt": "What is the sum of 2 and 3?",
# "name": "fn_add_numbers",
# "parameters": {"a": 2.0, "b": 3.0}
# },)
from typing import List
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionCallResult, UserPrompt
import json


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.llm = Small_LLM_Model(model_name=model_name)

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
            "You are a helpful assistant that only responds "
            "with JSON function calls.\n"
            "You must use one of these available functions:\n"
            f"{tools_desc}"
            "Output ONLY a JSON object with this exact structure:\n"
            '{"prompt": "original prompt", '
            '"name": "function_name", "parameters": {"arg": value}}\n'
            "No conversational text. No explanation. "
            "No text before or after the JSON."
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
        for _ in range(200): # Sécurité : max 200 tokens pour éviter l'infini
            # On concatène le prompt original et ce qu'on a déjà généré
            current_sequence = input_ids + generated_tokens
            
            # Le SDK nous donne les scores de probabilité (logits) pour le token suivant
            logits = self.llm.get_logits_from_input_ids(current_sequence)
            
            # --- C'est ici que tu devras filtrer les logits plus tard ---
            # Pour l'instant, on prend simplement le plus probable (le plus gros score)
            next_token_id = logits.index(max(logits))
            
            # Si le modèle génère le token "End Of String", il a fini sa phrase
            if next_token_id == self.llm._tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id)

        # 3. Finalisation
        # On transforme la liste de nombres en texte JSON
        raw_json_str = self.llm.decode(generated_tokens)

        #à protéger
        data_dict = json.loads(raw_json_str)
        return data_dict

            
        #     # 3. On crée l'objet Pydantic à partir du dictionnaire
        #     # Le **data_dict "déballe" les clés pour remplir les champs de la classe
        #     return FunctionCallResult(**data_dict)
            
        # except (json.JSONDecodeError, ValueError) as e:
        #     # Si l'IA a mal écrit le JSON ou s'il manque des champs Pydantic
        #     # On peut soit lever une erreur, soit gérer un cas de repli
        #     print(f"Erreur de parsing : {e}")
        #     raise


        # On transforme ce texte en objet Pydantic (FunctionCallResult)
        # return self._parse_result(raw_json_str, prompt_data.prompt)


# Étapes : 1. Appeler _build_system_prompt.
# 2. Transformer le texte en IDs avec self.llm.encode().
# 3. Lancer la boucle while ou for de génération.
# 4. Appeler _apply_constraints à chaque tour.
        pass

    def _apply_constraints(logits: List[float], current_text: str, functions: List[FunctionDefinition]) -> List[float]:
        pass
# Rôle : Le filtre de sécurité.Détail : Analyse le texte déjà généré (current_text).
# Si on vient d'écrire "name": ", cette fonction doit mettre à $-\infty$ tous les tokens qui ne sont pas des noms de fonctions valides.

    def _clean_output(raw_output: str) -> str:
        pass
# Rôle : Nettoyage de secours.
# Détail : Supprimer les espaces inutiles ou les caractères de fin de chaîne que le LLM aurait pu ajouter malgré la contrainte.

    def _parse_and_validate(json_str: str, original_prompt: str) -> FunctionCallResult:
        pass
# Rôle : La preuve par Pydantic.
# Détail : Utiliser json.loads() puis passer le dictionnaire à ton modèle FunctionCallResult(**data). Si ça plante ici, c'est que ta Phase 2 n'était pas assez stricte !