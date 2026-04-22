import sys
import argparse
from src.constrained_decoder import ConstrainedDecoder
from src.load import Loader
from src.prompt_processor import PromptProcessor
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to functions definition file")
    parser.add_argument(
        "-i",
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to user prompts file"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to output file or directory"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model name or path (e.g., 'HuggingFaceTB/SmolLM2-360M')"
    )
    
    args = parser.parse_args()

    loader = Loader()

    output_file_path = Path(args.output)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    sys.stdout.write("\033[1m--- CALL ME MAYBE ---\033[0m\n\n")
    sys.stdout.flush()

    try:
        functions = loader.get_functions(args.functions_definition)
        prompts = loader.get_prompts(args.input)

        sys.stdout.write(f"Input: {len(prompts)} prompts loaded.\n")
        sys.stdout.flush()

    except Exception as e:
        sys.stderr.write(f"\033[91m[ERROR]\033[0m {e}\n")
        sys.stderr.flush()
        sys.exit(1)

    try:
        sys.stdout.write("Initializing LLM ({args.model})...")
        sys.stdout.flush()

        constrained_decoder = ConstrainedDecoder(model_name=args.model)

        sys.stdout.write(" Done.\n")
        sys.stdout.flush()

    except Exception as e:
        sys.stderr.write(f"\n\033[91m[ERROR]\033[0m {e}\n")
        sys.stderr.flush()
        sys.exit(1)

    if prompts:
        try:
            sys.stdout.write("\n\033[1mStarting Real-Time Generation:"
                             "\033[0m\n")
            sys.stdout.write("-" * 50 + "\n")
            sys.stdout.flush()

            processor = PromptProcessor(constrained_decoder)
            processor.run(functions, prompts, output_file_path)

            sys.stdout.write(f"\n\033[92m[COMPLETED]\033[0m "
                             f"All results saved to: {output_file_path}\n")
            sys.stdout.flush()

        except Exception as e:
            sys.stdout.write(f"\n\033[91m[CRITICAL ERROR "
                             f"during generation]\033[0m {e}\n")
            sys.stdout.flush()
    else:
        sys.stdout.write("\033[93m[INFO]\033[0m "
                         "No prompts found to process.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
