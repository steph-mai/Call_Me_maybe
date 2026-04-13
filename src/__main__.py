# import argparse
import sys
from src.load import Loader
# from src.generator import Generator


def main():
    loader = Loader()

    try:
        data = loader.load_file("data/input/functions_definition.json")
        print(data)
    except Exception as e:
        print(f"\033[91m[ERROR]\033[0m {e}", file=sys.stderr)
        sys.exit(1)
    # 1. Configuration des arguments
    # avec argparse
    # parser = argparse.ArgumentParser(description="LLM Function Calling Program")
    
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
    
    # # Ici tu appelleras ton Loader avec args.input, args.functions_definition, etc.
    # # Puis ton Generator...
    
if __name__ == "__main__":
    main()