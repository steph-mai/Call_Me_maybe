import sys
import os
from src.constrained_decoder import ConstrainedDecoder
from src.load import Loader

def main():
    # 1. Nettoyage initial du terminal pour le confort visuel
    os.system('cls' if os.name == 'nt' else 'clear')
    
    loader = Loader()
    output_file_path = "data/output/function_calling_results.json"

    print("\033[1m--- CALL ME MAYBE: HIGH PERFORMANCE FSM ---\033[0m\n", flush=True)

    # 2. Chargement des données
    try:
        functions = loader.get_functions("data/input/functions_definition.json")
        prompts = loader.get_prompts("data/input/function_calling_tests.json")
        print(f"[*] Input: {len(prompts)} prompts loaded.", flush=True)
    except Exception as e:
        print(f"\033[91m[ERROR]\033[0m {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # 3. Initialisation (Le modèle et le vocabulaire sont chargés ici)
    try:
        print("[*] Initializing LLM & Vocab Mapping...", end="", flush=True)
        constrained_decoder = ConstrainedDecoder()
        print(" Done.", flush=True)
    except Exception as e:
        print(f"\n\033[91m[ERROR]\033[0m {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # 4. Lancement du Pipeline
    if prompts:
        try:
            print("\n\033[1mStarting Real-Time Generation:\033[0m", flush=True)
            print("-" * 50, flush=True)
            
            # La méthode .run() va maintenant imprimer chaque token en couleur
            constrained_decoder.run(
                functions=functions, 
                callables=prompts, 
                output_path=output_file_path
            )

            print("\n" + "-" * 50, flush=True)
            print(f"\n\033[92m[COMPLETED]\033[0m All results saved to: {output_file_path}", flush=True)
            
        except Exception as e:
            print(f"\n\033[91m[CRITICAL ERROR during generation]\033[0m {e}", flush=True)
    else:
        print("\033[93m[INFO]\033[0m No prompts found to process.", flush=True)

if __name__ == "__main__":
    main()
# # import argparse
# import sys
# import os
# import json
# from src.load import Loader
# from src.constrained_decoder import ConstrainedDecoder


# def main() -> None:

    
#     # if prompts:
#     #     try:
#     #         for i, prompt in enumerate(prompts, 1):
#     #             print(f"\n--- Test {i} ---")
#     #             result = ConstrainedDecoder.generate(prompt, functions)
                
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
    # # Puis ton constrained_decoder...
