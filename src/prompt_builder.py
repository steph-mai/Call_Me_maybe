class PromptBuilder:
    def __init__(self):
        self.system_instruction = "You are a precise data extraction agent."

    def build_name_selector_prompt(self, query: str, functions: list) -> str:
        # (Gardé identique à ta version, c'est l'étape 1)
        tools_desc = ""
        for f in functions:
            tools_desc += f"- {f['name']}: {f['description']}\n"
        return f"### SYSTEM\n{self.system_instruction}\n\n### TOOLS\n{tools_desc}\n### QUERY\n'{query}'\n\nTool Name: "

    def build_param_extractor_prompt(self, query: str, fn_info: dict) -> str:
        """
        ÉTAPE 2 : On guide le format selon le TYPE.
        """
        fn_name = fn_info.get("name")
        params = fn_info.get("parameters", {})
        
        # On construit une ligne d'exemple dynamique
        guide_parts = []
        for p_name, p_info in params.items():
            p_type = p_info.get("type", "string")
            if p_type == "number":
                guide_parts.append(f'"{p_name}": <number>')
            else:
                guide_parts.append(f'"{p_name}": "<string>"')
        
        format_guide = ", ". join(guide_parts)

        return (
            f"### QUERY\n'{query}'\n"
            f"### SELECTED TOOL\n{fn_name}\n"
            f"### INSTRUCTION\nExtract parameters in this format: {{{format_guide}}}\n\n"
            f"### RESPONSE\n"
            f"Parameters: {{"
        )