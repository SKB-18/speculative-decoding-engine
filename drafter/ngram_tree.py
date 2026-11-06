"""
Dynamic N-Gram Tree Drafter.
Discovers repetitive patterns and structured continuations from prompt context
and generation history to construct branching candidate trees.
"""

from typing import List, Optional, Dict
import torch

from .base import BaseDrafter
from tree.node import SpeculativeTree
from tree.builder import TreeBuilder


class NGramTreeDrafter(BaseDrafter):
    """
    Non-parametric n-gram speculative drafter.
    Finds occurrences of the recent token history in earlier context
    and extracts multi-path candidate continuations.
    """

    def __init__(
        self,
        min_ngram: int = 2,
        max_ngram: int = 5,
        max_branch_depth: int = 4,
        max_branches_per_match: int = 3,
    ):
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.max_branch_depth = max_branch_depth
        self.max_branches_per_match = max_branches_per_match

    def generate_tree(
        self,
        context_tokens: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        max_nodes: int = 16,
    ) -> SpeculativeTree:
        """
        Scans context_tokens for matching suffix n-grams and builds a SpeculativeTree.
        """
        if isinstance(context_tokens, torch.Tensor):
            tokens = context_tokens.flatten().tolist()
        else:
            tokens = list(context_tokens)

        L = len(tokens)
        if L < self.min_ngram:
            return SpeculativeTree()

        candidate_paths: List[List[int]] = []

        # Try largest n-grams first down to min_ngram
        for n in range(min(self.max_ngram, L - 1), self.min_ngram - 1, -1):
            target_ngram = tuple(tokens[-n:])

            # Search earlier in the sequence for target_ngram (up to L - n)
            matches: List[int] = []
            for i in range(L - n):
                if tuple(tokens[i : i + n]) == target_ngram:
                    matches.append(i + n)

            if matches:
                # For each match, extract continuation up to max_branch_depth
                for m_idx in matches[: self.max_branches_per_match]:
                    continuation = tokens[m_idx : m_idx + self.max_branch_depth]
                    if continuation and continuation not in candidate_paths:
                        candidate_paths.append(continuation)

                # Stop searching smaller n-grams if we found strong matches
                if len(candidate_paths) >= 2:
                    break

        if not candidate_paths:
            return SpeculativeTree()

        tree = TreeBuilder.from_candidate_paths(candidate_paths)
        if len(tree) > max_nodes:
            tree = TreeBuilder.prune_tree(tree, max_nodes=max_nodes)

        return tree
