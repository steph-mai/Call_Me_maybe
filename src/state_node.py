from typing import Dict, List


class StateNode:
    # Le nom Trie vient du mot anglais retrieval (récupération).
    # Le but est de récupérer une information très vite en suivant un chemin tracé.
    """
    Represents a single state in a Prefix Trie used for constrained decoding.

    This class is the building block of a Finite State Machine (FSM) that
    restricts the LLM's vocabulary during function name generation. It ensures
    that the model can only produce tokens that follow a valid path toward
    a predefined function name.

    Attributes:
        children (Dict[int, 'StateNode']): A mapping of token IDs to the next
            possible states. This acts as the "allowed vocabulary" for the
            current generation step.
        is_terminal (bool): Flag indicating if the current state represents
            the end of a complete and valid function name.
        name (str): The full string representation of the function name,
            stored only at terminal nodes for easy retrieval.
    """
    def __init__(self):
        self.children: Dict[int, 'StateNode'] = {}
        self.is_terminal: bool = False
        self.name: str = ""

    def insert_name(self, name_ids: List[int], name: str):
        """
        Inserts a function name into the state machine.

        This method traces a path of token IDs through the trie, creating
        new StateNodes as needed. The final node in the sequence is marked
        as terminal.

        Args:
            name_ids (List[int]): The sequence of token IDs representing
                the function name (provided by the tokenizer).
            name (str): The full text name of the function to be stored
                at the terminal node.
        """
        state_node = self
        for name_token_id in name_ids:
            if name_token_id not in state_node.children:
                state_node.children[name_token_id] = StateNode()
            state_node = state_node.children[name_token_id]
        state_node.is_terminal = True
        state_node.name = name
