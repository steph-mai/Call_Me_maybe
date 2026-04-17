from typing import List
import json
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionCallResult, UserPrompt
from src.constraints import JSONStructureEnforcer
from src.prompt_builder import PromptBuilder
from src.output_parser import OutputParser


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """
        initializes a generator capable of forcing an LLM to produce Json
        Parameters:
        an instance of an llm_sdk object that communicates with the llm,
        an instance of the JSONStructureEnforcer object
        responsible for constrained decoding
        """
        self.llm = Small_LLM_Model(model_name=model_name)

        # A un intérêt si on utilise des LLM presentant des formats
        # differents de vocabulaire (ID: token) ou (token: ID)
        # voca_path = self.llm.get_path_to_vocab_file()
        # with open(voca_path, mode='r', encoding='utf-8') as f:
        #     raw_vocab = json.load(f)
        # vocab = {}
        # for key, value in raw_vocab.items():
        #     try:
        #         v_id = int(key)
        #         v_token = str(value)
        #     except ValueError:
        #         v_id = int(value)
        #         v_token = str(key)
        #     vocab[v_id] = v_token
        # if not vocab:
        #     raise ValueError(
        #         "Empty vocabulary. Check your vocabulary file"
        #         )
        self.prompt_builder = PromptBuilder()
        self.output_parser = OutputParser()
        self.stucture_enforcer = JSONStructureEnforcer(self.llm.get_vocab(), self.llm)
        
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
        # [[1, 54, 342, 12, ...]]. prompt_ids (le Tensor complet) : [[...]]
        # prompt_ids[0] (le premier élément) : [...]. tolist transforme le 
        # tenseur en liste
        prompt_ids = self.llm.encode(prompt_str).tolist()[0]

        max_new_tokens = 512
        generated_tokens: list[int] = []

        generated_text = ""

        while len(generated_tokens) < max_new_tokens:
            if generated_text.strip().endswith('}'):
                # PQ
                if (
                    generated_text.count('{') == generated_text.count('}')
                    and generated_text.count('{') > 0
                ):
                    break

            raw_logits = self.llm.get_logits_from_input_ids(
                prompt_ids + generated_tokens
                )

            constrained_logits = self.stucture_enforcer.enforce_constraints(raw_logits, generated_text)

            max_score = max(constrained_logits)
            if max_score <= -1e10:
                raise RuntimeError("Dead-end reached: The constraints filtered out ALL possible tokens.")

            next_token_id = constrained_logits.index(max_score)

            if next_token_id == self.llm._tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)

            next_token_text = self.llm.decode([next_token_id])
            generated_text += next_token_text
            # Le problème du "Buffer" : Normalement, pour économiser des ressources,
            # Python attend d'avoir "beaucoup" de texte à afficher avant de l'envoyer
            # réellement à l'écran (buffering).
            # L'action : flush=True force Python à vider son tampon et à envoyer
            # le caractère au terminal immédiatement.
            print(next_token_text, end="", flush=True)

        return self.output_parser.output_parse(generated_text, prompt_data)
