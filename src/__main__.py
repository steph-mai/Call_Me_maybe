# import argparse
import sys
from src.load import Loader
from src.generator import Generator

def main() -> None:
    loader = Loader()

    # 1. Chargement des données
    try:
        functions = loader.get_functions("data/input/functions_definition.json")
        prompts = loader.get_prompts("data/input/function_calling_tests.json")
        print(f"\033[94m[INFO]\033[0m Données chargées : {len(functions)} fonctions, {len(prompts)} tests.")
    except Exception as e:
        print(f"\033[91m[ERROR Loader]\033[0m {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Initialisation du modèle (Une seule fois !)
    try:
        print("\033[94m[INFO]\033[0m Chargement du modèle LLM en cours...")
        generator = Generator()
    except Exception as e:
        print(f"\033[91m[ERROR Model]\033[0m {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Test de génération sur le PREMIER prompt uniquement
    if prompts:
        try:
            print(f"\n\033[94m[EXECUTION]\033[0m Test sur : {prompts[0].prompt}")
            result = generator.generate(prompts[0], functions)
            
            # print(f"\n\033[92m[RESULTAT FINAL]\033[0m\n{result}")
        except Exception as e:
            print(f"\n\033[93m[WARNING Generation]\033[0m {e}")

if __name__ == "__main__":
    main()


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
