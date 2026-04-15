from typing import List


class JSONLogitsProcessor:
    def __init__(self, vocab: dict):
        self.vocab = vocab

    def apply_constraints(self, logits: List[float], current_text: str) -> List[float]:
        """
        It modifies the logits returned by the LLM to retain those 
        that can be used to produce JSON. 
        It operates on the principle of a state machine.
        """
        # Si on est au tout début
        if not current_text.strip():
            return self._mask_for_start(logits)

        # TODO à compléter
        # elif current_text.endswith('{'):
        #     return self._mask_for_prompt_key(logits)

        return logits

    def _mask_for_start(self, logits: List[float]) -> List[float]:
        new_logits = [-1e10] * len(logits)
        for tid, t_content in self.vocab.items():
            if str(t_content).strip().startswith('{'):
                new_logits[tid] = logits[tid]
        return new_logits