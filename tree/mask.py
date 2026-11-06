"""
Causal Tree Attention Mask Generator.
Constructs 2D and 3D attention masks enforcing ancestor-only causal dependency
to eliminate cross-branch information leakage during single-pass parallel verification.
"""

from typing import Optional, Union
import torch
from .node import SpeculativeTree


def build_tree_attention_mask(
    tree: SpeculativeTree,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    as_bool: bool = False,
) -> torch.Tensor:
    """
    Constructs an [N, N] causal attention mask for tree candidate nodes.
    
    If as_bool is False:
        mask[i, j] = 0.0 if j is ancestor of i (or j == i)
        mask[i, j] = -inf otherwise
    If as_bool is True:
        mask[i, j] = True if j is ancestor of i (or j == i)
        mask[i, j] = False otherwise
    """
    N = len(tree)
    if N == 0:
        if as_bool:
            return torch.empty((0, 0), dtype=torch.bool, device=device)
        return torch.empty((0, 0), dtype=dtype, device=device)

    if as_bool:
        mask = torch.zeros((N, N), dtype=torch.bool, device=device)
        for i in range(N):
            ancestors = tree.get_ancestors(i, include_self=True)
            for j in ancestors:
                mask[i, j] = True
        return mask

    mask = torch.full((N, N), float("-inf"), dtype=dtype, device=device)
    for i in range(N):
        ancestors = tree.get_ancestors(i, include_self=True)
        for j in ancestors:
            mask[i, j] = 0.0

    return mask


def build_full_speculative_attention_mask(
    tree: SpeculativeTree,
    prefix_len: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    as_bool: bool = False,
) -> torch.Tensor:
    """
    Constructs the full [(prefix_len + N), (prefix_len + N)] attention mask
    combining causal prefix attention with tree speculative candidate attention.

    Topology:
    ┌──────────────────────┬─────────────────────────┐
    │  Prefix -> Prefix    │   Prefix -> Tree        │
    │  (Standard Causal)   │   (All -inf / False)    │
    ├──────────────────────┼─────────────────────────┤
    │  Tree -> Prefix      │   Tree -> Tree          │
    │  (All 0.0 / True)    │   (Ancestor Causal)     │
    └──────────────────────┴─────────────────────────┘
    """
    N = len(tree)
    total_len = prefix_len + N

    if as_bool:
        full_mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
        # 1. Prefix -> Prefix: lower triangular (causal)
        full_mask[:prefix_len, :prefix_len] = torch.tril(
            torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=device)
        )
        # 2. Prefix -> Tree: already False (no future leakage)
        # 3. Tree -> Prefix: tree candidates attend to all prefix tokens
        full_mask[prefix_len:, :prefix_len] = True
        # 4. Tree -> Tree: ancestor attention
        if N > 0:
            tree_mask = build_tree_attention_mask(tree, as_bool=True, device=device)
            full_mask[prefix_len:, prefix_len:] = tree_mask
        return full_mask

    full_mask = torch.full((total_len, total_len), float("-inf"), dtype=dtype, device=device)

    # 1. Prefix -> Prefix: causal lower-triangular 0.0
    prefix_tril = torch.tril(torch.ones((prefix_len, prefix_len), device=device))
    full_mask[:prefix_len, :prefix_len] = torch.where(
        prefix_tril == 1,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )

    # 2. Prefix -> Tree: stays -inf

    # 3. Tree -> Prefix: 0.0 (all tree tokens attend to all prefix tokens)
    full_mask[prefix_len:, :prefix_len] = 0.0

    # 4. Tree -> Tree: ancestor causal mask
    if N > 0:
        tree_mask = build_tree_attention_mask(tree, dtype=dtype, device=device, as_bool=False)
        full_mask[prefix_len:, prefix_len:] = tree_mask

    return full_mask
