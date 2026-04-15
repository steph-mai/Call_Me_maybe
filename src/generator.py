from typing import List
import json
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionCallResult, UserPrompt
from src.constraints import JSONLogitsProcessor
from src.prompt_builder import PromptBuilder
from src.output_parser import OutputParser


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """
        initializes a generator capable of forcing an LLM to produce Json
        Parameters:
        an instance of an llm_sdk object that communicates with the llm,
        an instance of the JSONLogitsProcessor object
        responsible for constrained decoding
        """
        self.llm = Small_LLM_Model(model_name=model_name)

        voca_path = self.llm.get_path_to_vocab_file()
        with open(voca_path, mode='r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
        vocab = {}
        for key, value in raw_vocab.items():
            try:
                v_id = int(key)
                v_token = str(value)
            except ValueError:
                v_id = int(value)
                v_token = str(key)
            vocab[v_id] = v_token
            if not vocab:
                raise ValueError(
                    "Empty vocabulary. Check your vocabulary file"
                    )
        self.logits_processor = JSONLogitsProcessor(vocab)
        self.prompt_builder = PromptBuilder()
        self.output_parser = OutputParser()

    def generate(self,
                 prompt_data: UserPrompt,
                 functions: List[FunctionDefinition]
                 ) -> FunctionCallResult:
        """
        Generates valid JSON by constraining the LLM.
        Uses the methods of the llm_sdk to transform the initial prompt
        into tokens, then applies the constraints by manipulating the logits
        and translates the logits into text using the llm_sdk.
        Returns a parsed and validated JSON (Pydantic object)
        """
        prompt_str = self.prompt_builder.prompt_build(prompt_data, functions)
        # Comme on n´envoie qu'un seul prompt_str, le résultat de encode()
        # est un Tensor (ou une liste de listes) de type :
        # [[1, 54, 342, 12, ...]]. input_ids (le Tensor complet) : [[...]]
        # input_ids[0] (le premier élément) : [...].
        input_ids = self.llm.encode(prompt_str).tolist()[0]

        max_new_tokens = 512
        generated_tokens: list[int | float] = []

        while len(generated_tokens) < max_new_tokens:
            logits = self.llm.get_logits_from_input_ids(
                input_ids + generated_tokens
                )
            current_text = self.llm.decode(generated_tokens)
            logits = self.logits_processor.apply_constraints(
                logits,
                current_text
                )
            next_token_id = logits.index(max(logits))
            if next_token_id == self.llm._tokenizer.eos_token_id:
                break
            generated_tokens.append(next_token_id)
# Le problème du "Buffer" : Normalement, pour économiser des ressources,
# Python attend d'avoir "beaucoup" de texte à afficher avant de l'envoyer
# réellement à l'écran (buffering).
# L'action : flush=True force Python à vider son tampon et à envoyer
# le caractère au terminal immédiatement.
            print(self.llm.decode([next_token_id]), end="", flush=True)

        raw_json_str = self.llm.decode(generated_tokens)
        return self.output_parser.output_parse(raw_json_str, prompt_data)
