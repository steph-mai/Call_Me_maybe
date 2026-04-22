from typing import Dict, List


class StateNode:
    """
    A node in a Prefix Trie used for constrained decoding.
    Each node represents a state in a Finite State Machine (FSM).
    """
    def __init__(self) -> None:
        self.children: Dict[int, 'StateNode'] = {}
        self.is_terminal: bool = False
        self.name: str = ""

    def insert_name(self, name_ids: List[int], name: str) -> None:
        "Builds the Trie path for a function name using its token IDs."
        state_node = self

        for name_token_id in name_ids:
            if name_token_id not in state_node.children:
                state_node.children[name_token_id] = StateNode()
            state_node = state_node.children[name_token_id]

        state_node.is_terminal = True

        state_node.name = name
