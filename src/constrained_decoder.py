import json
import re
import numpy as np
from typing import Any, List, Set, Dict, cast
from llm_sdk import Small_LLM_Model  # type: ignore
from src.models import FunctionDefinition
from src.state_node import StateNode

NEG_INF: float = -1e11


class ConstrainedDecoder:
    """
    Handles constrained decoding logic by masking logits
    to force the LLM to follow specific schemas or types.
    """
    def __init__(self, model_name: str) -> None:
        """
        Initializes the decoder by loading the model vocabulary and
        pre-filtering token sets for numeric and boolean constraints.
        """
        self.llm = Small_LLM_Model(model_name=model_name)

        self.vocab = self._load_vocab()

        self._tokens_num = self._build_set(r'^[0-9.\-eE]+$')
        self._tokens_stop = self._build_set(r'^[,\}\]:\s\n\t]+$')
        self.tokens_boolean = self._build_set(r'^(True|False)$')
        self.quote_id = self.llm.encode('"')[0].tolist()[-1]

    def _load_vocab(self) -> dict:
        """
        Loads vocabulary from JSON file (Qwen)
        or from internal tokenizer (e.g., SmollM2).
        """
        try:
            with open(
                self.llm.get_path_to_vocab_file(), "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
                # json.load returns Any
                # Use cast to transform Any en Dict
                return cast(Dict[str, int], data)
        except Exception:
            return cast(Dict[str, int], self.llm.tokenizer.get_vocab())

    def _build_set(self, pattern: str) -> Set[int]:
        """Filters vocabulary to find token IDs matching a regex pattern."""
        allowed = set()
        for tok_str, tid in self.vocab.items():
            # Clean control characters and model-specific prefixes
            # (Ġ ou \u0120)
            clean = re.sub(r'[\s\u0120Ġ]+', '', tok_str)

            if clean and re.match(pattern, clean):
                allowed.add(tid)
        return allowed

    def _get_masked_next(
            self,
            ids: List[int],
            allowed: Set[int] | None
            ) -> int:
        """
        Calculates the next token ID by applying a mask to logits.
        Only allowed tokens retain their original logit values.
        """
        logits = np.array(
            self.llm.get_logits_from_input_ids(ids),
            dtype=np.float32
        )
        # The model returns a tensor of scores. We use 'shape'
        # to check its dimensions and isolate the 1D vector (logits)
        # of the last token for the next prediction.
        while len(logits.shape) > 1:
            logits = logits[-1]

        # Apply the mask: default to negative infinity
        if allowed is not None:
            mask = np.full(logits.shape, NEG_INF, dtype=np.float32)

            # len(mask) is the vocabulary size. This check ensures the token ID
            # is within the valid range of the model's index.
            for tid in allowed:
                if tid < len(mask):
                    mask[tid] = logits[tid]
            return int(np.argmax(mask))

        return int(np.argmax(logits))

    def force_name(self,
                   full_prompt_ids: List[int],
                   functions: List[FunctionDefinition]
                   ) -> str:
        """
        Forces the model to select a valid function name by
        traversing a prefix tree (Trie) of available function names.
        """
        trie_root = StateNode()

        # Initialize the prefix tree with all possible function names
        for f in functions:
            name_tokens = self.llm.encode(f.name)[0].tolist()
            trie_root.insert_name(name_tokens, f.name)

        current_node = trie_root
        # sequence starts with the prompt and grows as tokens are predicted
        prompt_sequence = list(full_prompt_ids)

        while True:
            # The allowed tokens are the children of the current node.
            allowed = set(current_node.children.keys())

            if not allowed:
                break

            # If multiple paths exist, mask the logits; otherwise,
            # take the only choice
            if len(allowed) > 1:
                chosen = self._get_masked_next(prompt_sequence, allowed)
            else:
                # Convert the set to an iterable and retrieve the first
                # (and only) element
                chosen = next(iter(allowed))

            prompt_sequence.append(chosen)

            current_node = current_node.children[chosen]

        return current_node.name

    def extract_param_value(self, ids: List[int], p_type: str) -> Any:
        """
        Extracts a single parameter value from the LLM based on its type.
        Supports number, boolean, and string extraction.
        """
        prompt_sequence = list(ids)
        extracted_value = ""

        if p_type == "number":
            allowed = self._tokens_num | self._tokens_stop
            max_tokens = 20
        elif p_type == "boolean":
            allowed = self.tokens_boolean | self._tokens_stop
            max_tokens = 5
        else:
            allowed = None
            max_tokens = 200

        for i in range(max_tokens):
            if p_type == "number" or p_type == "boolean":
                chosen = self._get_masked_next(prompt_sequence, allowed)

            else:
                logits = np.array(
                    self.llm.get_logits_from_input_ids(prompt_sequence),
                    dtype=np.float32
                )
                # Reduce the tensor (Batch, Sequence, Vocab) to a 1D vector
                # of the last token's raw scores
                while len(logits.shape) > 1:
                    logits = logits[-1]

                # Prevent immediate closing of string
                if i == 0:
                    logits[self.quote_id] = NEG_INF
                chosen = int(np.argmax(logits))

            # Stop if a termination token is found
            is_stop_token = (
                (chosen == self.quote_id) or
                ((p_type == "number" or p_type == "boolean") and
                 chosen in self._tokens_stop)
            )
            if is_stop_token:
                if p_type == "number" and extracted_value:
                    if (
                        "." not in extracted_value
                        and "e" not in extracted_value.lower()
                    ):
                        extracted_value += ".0"
                break

            decoded_token = self.llm.decode([chosen])

            # Safety break to prevent hallucinations
            if "\n" in decoded_token and i > 0:
                break

            extracted_value += decoded_token
            prompt_sequence.append(chosen)

        return extracted_value.strip()
