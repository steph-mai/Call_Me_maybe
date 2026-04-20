class PromptBuilder:
    def __init__(self):
        self.system_instruction = "You are a precise data extraction agent."

    def build_name_selector_prompt(self, query: str, functions: list) -> str:
        functions_desc = ""
        for f in functions:
            # On supporte les objets Pydantic ou les dictionnaires
            name = f.name if hasattr(f, 'name') else f['name']
            desc = f.description if hasattr(f, 'description') else f['description']
            functions_desc += f"- {name}: {desc}\n"
            
        return (f"### SYSTEM\n{self.system_instruction}\n\n"
                f"### TOOLS\n{functions_desc}\n"
                f"### QUERY\n'{query}'\n\n"
                f"Tool Name: ")

    def build_param_extractor_prompt(self, query: str, fn_info: dict) -> str:
        fn_name = fn_info.get("name")
        params = fn_info.get("parameters", {})
        
        guide_parts = []
        specific_instructions = []
        
        # 1. Construction des règles de formatage pour TOUS les paramètres
        for p_name, p_info in params.items():
            # Support Pydantic (p_info.type) ou Dict (p_info['type'])
            p_type = p_info.type if hasattr(p_info, 'type') else p_info.get("type", "string")
            
            if p_type == "number":
                guide_parts.append(f'"{p_name}": <number>')
                specific_instructions.append(f"- '{p_name}' must be a numeric value (no quotes).")
            else:
                guide_parts.append(f'"{p_name}": "<text>"')
                specific_instructions.append(f"- '{p_name}' must be a string wrapped in double quotes.")

        format_guide = ", ".join(guide_parts)
        joined_instructions = "\n".join(specific_instructions)

        # 2. Gestion spécifique du Few-Shot pour la Regex (Indépendante de la boucle)
        example_block = ""
        if fn_name == "fn_substitute_string_with_regex":
            example_block = (
                "### EXAMPLE\n"
                "Query: Replace 'apple' with 'banana' in 'I like apple'\n"
                "Parameters: {\"source_string\": \"I like apple\", \"regex\": \"apple\", \"replacement\": \"banana\"}\n\n"
            )

        # 3. Retour du prompt final harmonisé
        return (
            f"### SYSTEM\n{self.system_instruction}\n\n"
            f"{example_block}"
            f"### QUERY\n'{query}'\n"
            f"### SELECTED TOOL\n{fn_name}\n"
            f"### FORMATTING RULES\n{joined_instructions}\n"
            f"### INSTRUCTION\nExtract parameters in this format: {{{format_guide}}}\n\n"
            f"### RESPONSE\n"
            f"Parameters: {{"
        )