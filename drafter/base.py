"""
Base Speculative Drafter Interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from tree.node import SpeculativeTree


class BaseDrafter(ABC):
    """
    Abstract interface for speculative draft tree generators.
    """

    @abstractmethod
    def generate_tree(
        self,
        context_tokens: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        max_nodes: int = 16,
    ) -> SpeculativeTree:
        """
        Generates a tree of candidate tokens based on token context and/or hidden states.

        Args:
            context_tokens: [B, S] or [S] token IDs of the current prompt + generated text.
            hidden_state: Optional [B, S, D] or [D] final hidden representation of the last token.
            max_nodes: Maximum allowable nodes in the candidate tree.

        Returns:
            SpeculativeTree containing branching candidate tokens.
        """
        pass
