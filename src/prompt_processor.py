import sys
import json
from typing import List, Any
from src.models import FunctionDefinition, FunctionCallResult, UserPrompt


class PromptProcessor:
    """
    Orchestrates the high-level workflow of turning a user prompt
    into a validated FunctionCallResult.
    """
    def __init__(self, decoder) -> None:
        self.decoder = decoder
        # dico qui va stocker en cache les prompts/reponses
        self._cache = {}

    def process(self,
                user_prompt: str,
                functions: List[FunctionDefinition],
                system_prompt_ids: List[int]
                ) -> FunctionCallResult:

        name_selection_ids = system_prompt_ids + self.decoder.llm.encode(
            f"\nQuery: {user_prompt}\nFunction:"
            )[0].tolist()
        selected_name = self.decoder.force_name(name_selection_ids, functions)

        sys.stdout.write(f"\n  [NAME]: \033[94m{selected_name}\033[0m")

        selected_function_def = None
        for f in functions:
            if f.name == selected_name:
                selected_function_def = f
                break
        if selected_function_def is None:
            raise ValueError(
                f"Error: The function '{selected_name}' does not exist."
                )

        final_params: dict[str, float | int | bool | str] = {}
        for p_name, p_info in selected_function_def.parameters.items():

            instruction = (
                f"\nTask: {user_prompt}\n"
                f"Function: {selected_name}\n"
                f"{p_name}="
            )
            if p_info.type == "string":
                instruction += '"'

            full_prompt_ids = (
                system_prompt_ids
                + self.decoder.llm.encode(instruction)[0].tolist()
            )

            raw_value = self.decoder.extract_param_value(
                full_prompt_ids,
                p_info.type)
            
            raw_value = self._advanced_recovery(p_name, p_info.type, raw_value)

            # if isinstance(raw_value, str):
            #     raw_value = raw_value.replace('"', '').strip()
            #     if p_name == "regex":
            #         while raw_value.count('(') > raw_value.count(')'):
            #             raw_value += ')'
            #         while raw_value.count('[') > raw_value.count(']'):
            #             raw_value += ']'
            #     if p_name == "replacement" and "**" in raw_value:
            #         raw_value = "*"

            # if p_info.type == "boolean":
            #     raw_value = True if raw_value.strip() == "True" else False

            final_params[p_name] = raw_value

            if p_info.type == "number":
                color = "\033[92m"
            elif p_info.type == "boolean":
                color = "\033[95m"
            else:
                color = "\033[93m"
            sys.stdout.write(f" | {p_name}: {color}{raw_value}\033[0m")
            sys.stdout.flush()

        return FunctionCallResult(
            prompt=user_prompt,
            name=selected_name,
            parameters=final_params
            )

    def run(
            self,
            functions: List[FunctionDefinition],
            user_prompts: List[UserPrompt],
            output_path: str
            ) -> None:
        # model_dump : methode de Pydantic qui prend toutes les donnees
        # stockees dans une instance de classe et les "decharge" dans un
        # format standars (dico)
        tools_list = [f.model_dump() for f in functions]
        # recupere les IDs des tokens du system_prompt qui comprend
        # les fonctions disponibles en format JSON
        # on recupere le premier element de la matrice renvoyee
        # (comme on envoie un seul
        # system prompt). on le transforme em liste
        # (souvent renvoie un tenseur py torch ou un tqblequ numpy)
        system_prompt_ids = self.decoder.llm.encode(
            f"System: Tool Extractor. "
            f"Tools: {json.dumps(tools_list)}"
            )[0].tolist()

        results: List[FunctionCallResult] = []
        for index, user_prompt in enumerate(user_prompts):
            prompt_text = user_prompt.prompt

            if prompt_text in self._cache:
                res_dict = self._cache[prompt_text]
                sys.stdout.write(f"\n[{index+1}] Prompt: {prompt_text[:30]}... (CACHED)")
                results.append(res_dict)
                continue
                
            sys.stdout.write(f"\n[{index+1}/{len(user_prompts)}] "
                             f"Prompt: {user_prompt.prompt[:50]}...")

            res: FunctionCallResult = self.process(
                user_prompt.prompt,
                functions,
                system_prompt_ids
            )

            res_dict = res.model_dump()
            self._cache[prompt_text] = res_dict

            results.append(res_dict)

        with open(output_path, "w", encoding="utf-8") as f:
            # Par défaut (True) : Python remplace tous les caractères
            # non-anglais par des codes ("é" devient \u00e9)
            # Avec False : Python écrit les caractères tels quels (en UTF-8).
            json.dump(results, f, indent=4, ensure_ascii=False)

    def _advanced_recovery(self, p_name: str, p_type: str, raw_value: Any) -> Any:
        """Centralise les mécanismes de récupération et de nettoyage."""
        
        # --- 1. Gestion des Booléens (Normalisation) ---
        if p_type == "boolean":
            # Accepte différentes formes de "vrai" au cas où le masque laisserait passer un token proche
            if str(raw_value).strip().lower() in ["true", "1", "yes", "y", "t"]:
                return True
            return False

        # --- 2. Réparation des chaînes (Regex & Artefacts) ---
        if isinstance(raw_value, str):
            # Nettoyage des guillemets et espaces
            val = raw_value.replace('"', '').strip()
            
            # Réparation des Regex (Parenthèses et Crochets)
            if p_name == "regex":
                for opening, closing in [('(', ')'), ('[', ']')]:
                    while val.count(opening) > val.count(closing):
                        val += closing
            
            # Correction des répétitions bizarres (ton fix sur les **)
            if "**" in val:
                val = val.replace("**", "*")
                
            return val
        
        if p_type == "number":
            try:
                return float(raw_value)
            except (ValueError, TypeError):
                # Si le float() échoue, on renvoie une valeur par défaut cohérente
                # plutôt que de laisser le programme planter.
                return 0.0

        return raw_value