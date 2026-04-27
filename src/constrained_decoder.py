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

        vocab = self._load_vocab()

        self._tokens_num = self._built_set_with_vocab(
            vocab, r'^[0-9.\-eE]+$'
            )
        self._tokens_stop = self._built_set_with_vocab(
            vocab, r'^[,\}\]:\s\n\t]+$'
            )
        self.tokens_boolean = self._built_set_with_vocab(
            vocab, r'^(True|False)$'
            )
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
                return cast(Dict[str, int], data)
        except Exception:
            return cast(Dict[str, int], self.llm.tokenizer.get_vocab())

    def _built_set_with_vocab(self, vocab: dict, pattern: str) -> Set[int]:
        """Filters vocabulary to find token IDs matching a regex pattern."""
        allowed = set()
        for tok_str, tid in vocab.items():
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
        while len(logits.shape) > 1:
            logits = logits[-1]

        if allowed is not None:
            mask = np.full(logits.shape, NEG_INF, dtype=np.float32)

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

        for f in functions:
            name_tokens = self.llm.encode(f.name)[0].tolist()
            trie_root.insert_name(name_tokens, f.name)

        current_node = trie_root
        prompt_sequence = list(full_prompt_ids)

        while True:
            allowed = set(current_node.children.keys())

            if not allowed:
                break

            if len(allowed) > 1:
                chosen = self._get_masked_next(prompt_sequence, allowed)
            else:
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
        is_numeric = (
            p_type == "number" or
            p_type == "float" or
            p_type == "integer"
        )
        is_boolean = p_type == "boolean"

        if is_numeric:
            allowed = self._tokens_num | self._tokens_stop
            max_tokens = 30
        elif is_boolean:
            allowed = self.tokens_boolean | self._tokens_stop
            max_tokens = 5
        else:
            allowed = None
            max_tokens = 200

        for i in range(max_tokens):
            if is_numeric or is_boolean:
                chosen = self._get_masked_next(prompt_sequence, allowed)

            else:
                logits = np.array(
                    self.llm.get_logits_from_input_ids(prompt_sequence),
                    dtype=np.float32
                )
                while len(logits.shape) > 1:
                    logits = logits[-1]

                if i == 0:
                    logits[self.quote_id] = NEG_INF
                chosen = int(np.argmax(logits))

            if p_type == "string":
                if chosen == self.quote_id:
                    break
            elif is_numeric or is_boolean:
                if chosen in self._tokens_stop:
                    break

            decoded_token = self.llm.decode([chosen])

            if "\n" in decoded_token and i > 0:
                break

            extracted_value += decoded_token
            prompt_sequence.append(chosen)

        final_str = extracted_value.strip()

        if p_type == "number":
            try:
                return float(final_str)
            except ValueError:
                return 0.0

        if p_type == "integer":
            try:
                return int(final_str)
            except ValueError:
                return 0

        if p_type == "boolean":
            return final_str.lower() in ["true", "1", "yes"]

        return final_str
