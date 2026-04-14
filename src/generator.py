# genere le format JSON attendu ({
# "prompt": "What is the sum of 2 and 3?",
# "name": "fn_add_numbers",
# "parameters": {"a": 2.0, "b": 3.0}
# },)
from typing import List
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionCallResult, PromptData


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.llm = Small_LLM_Model(model_name=model_name)

    def _build_full_prompt(self,
                           prompt_data: PromptData,
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

        # 2. Construction du System Prompt
        system_prompt = (
            "You are a helpful assistant that calls functions to answer questions.\n"
            "Available tools:\n"
            f"{tools_desc}"
            "Output ONLY a JSON object with this exact structure:\n"
            '{"prompt": "original prompt", "name": "function_name", "parameters": {"arg": value}}\n'
            "No prose. No explanation. No text before or after the JSON."
        )

        # 3. Formatage final (Template Qwen/Chat)
        full_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_data.prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return full_text
# Rôle : Créer le mode d'emploi.
# Détail : Doit lister chaque fonction, sa description et ses paramètres avec 
# leurs types.
# Instruction cruciale : Préciser au modèle qu'il ne doit répondre que par du 
# JSON, sans texte avant ou après.

    def _format_prompt(user_query: str, system_prompt: str) -> str:
# Rôle : Fusionner les instructions et la question de l'utilisateur 
# selon le format attendu par le modèle (souvent un format Chat
# comme user: ... assistant: ...).
        pass

    def generate(prompt_data: PromptData, functions: List[FunctionDefinition]) -> FunctionCallResult:
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