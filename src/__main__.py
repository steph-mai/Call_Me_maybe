from src.generator import Generator
import json


def main():
    list_functions = [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together and return their sum.",
        },
        {
            "name": "fn_greet",
            "description": "Generate a greeting message for a person by name.",
        }
    ]

    user_queries = [
        "What is the sum of 2 and 3?",
        "What is the sum of 265 and 345?",
        "Greet shrek"
    ]

    print("--- Initializing Generator (Loading Model & Vocab) ---")
    generator = Generator(model_name="Qwen/Qwen3-0.6B")

    final_results = []

    for i, query in enumerate(user_queries, 1):
        print(f"\n[Test {i}] Query: '{query}'")
        print("Output: ", end="")
        
        try:
            result = generator.generate(query, list_functions)
            final_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to process query '{query}': {e}")

    print("FINAL STRUCTURED OUTPUT")
    print(json.dumps(final_results, indent=2))

if __name__ == "__main__":
    main()

# # import argparse
# import sys
# import os
# import json
# from src.load import Loader
# from src.generator import Generator


# def main() -> None:
#     loader = Loader()
#     output_file_path = "data/output/function_calling_results.json"

#     try:
#         functions = loader.get_functions("data/input/functions_definition.json")
#         prompts = loader.get_prompts("data/input/function_calling_tests.json")
#     except Exception as e:
#         print(f"\033[91m[ERROR Loader]\033[0m {e}", file=sys.stderr)
#         sys.exit(1)

#     try:
#         generator = Generator()
#     except Exception as e:
#         print(f"\033[91m[ERROR Model]\033[0m {e}", file=sys.stderr)
#         sys.exit(1)

#     all_results = []

#     if prompts:
#         try:
#             for i, prompt in enumerate(prompts[1:2], 1):
#                 print(f"\n--- Test {i} ---")

#                 result = generator.generate(prompt, functions)
#     # model_dump prend les données stockées dans une instance de classe
#     # et les décharge dans un format standard (dico)
#                 data_to_save = result.model_dump()
#                 all_results.append(data_to_save)
#     # os.path.dirname : Récupère la partie "dossier" de ton chemin.
#     # os.makedirs : Crée toute l'arborescence (si data n'existe pas, 
#     # il crée data, puis output).
#     # exist_ok=True : Pas de pb si le dir existe déjà
#             directory = os.path.dirname(output_file_path)
#             if directory:
#                 os.makedirs(directory, exist_ok=True)

#             with open(output_file_path, "w", encoding="utf-8") as f:
#                 # indent=4 : Sans cela, tout le JSON sera sur une seule ligne.
#                 json.dump(all_results, f, indent=4)
#             print(f"\033[92m[SUCCESS]\033[0m Generated File: {output_file_path}")

#         except Exception as e:
#             print(f"\n\033[93m[WARNING Generation]\033[0m {e}")
    
    
#     # if prompts:
#     #     try:
#     #         for i, prompt in enumerate(prompts, 1):
#     #             print(f"\n--- Test {i} ---")
#     #             result = generator.generate(prompt, functions)
                
#     #             # Affichage version JSON propre dans le terminal
#     #             print("\n[RESULTAT JSON] :")
#     #             print(result.model_dump_json(indent=4))
    

#         except Exception as e:
#             print(f"\n\033[93m[WARNING Generation]\033[0m {e}")

# if __name__ == "__main__":
#     main()


    # 1. Configuration des arguments
    # avec argparse
    # parser = argparse.ArgumentParser
    # (description="LLM Function Calling Program")

    # parser.add_argument("--functions_definition",
    #                     default="data/definitions.json", # Chemin par défaut
    #                     help="Path to functions definition JSON")

    # parser.add_argument("--input",
    #                     default="data/input/", # Dossier par défaut
    #                     help="Path to input JSON or directory")

    # parser.add_argument("--output",
    #                     default="data/output/", # Dossier par défaut
    #                     help="Path to output directory or file")

    # args = parser.parse_args()

    # # 2. Orchestration
    # print(f"Lancement avec l'entrée : {args.input}")

    # # Ici tu appelleras ton Loader avec args.input, args.
    # functions_definition, etc.
    # # Puis ton Generator...
