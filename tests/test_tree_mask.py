"""
Unit tests for Tree-Attention Mask Generation and Causal Isolation.
"""

import pytest
import torch
from tree.node import SpeculativeTree
from tree.mask import build_tree_attention_mask, build_full_speculative_attention_mask


def test_empty_tree_mask():
    tree = SpeculativeTree()
    mask = build_tree_attention_mask(tree)
    assert mask.shape == (0, 0)

    bool_mask = build_tree_attention_mask(tree, as_bool=True)
    assert bool_mask.shape == (0, 0)


def test_tree_attention_mask_ancestor_dependency():
    tree = SpeculativeTree()
    # 0 -> 1 -> 2
    #   -> 3
    tree.add_node(token_id=10, depth=0)  # 0
    tree.add_node(token_id=20, depth=1, parent_id=0)  # 1
    tree.add_node(token_id=30, depth=2, parent_id=1)  # 2
    tree.add_node(token_id=40, depth=1, parent_id=0)  # 3

    mask = build_tree_attention_mask(tree, as_bool=True)
    assert mask.shape == (4, 4)

    # Diagonal is always True (self-attention)
    for i in range(4):
        assert bool(mask[i, i]) is True

    # Node 1 attends to Node 0 (parent)
    assert bool(mask[1, 0]) is True
    # Node 0 does NOT attend to Node 1 (no future lookahead)
    assert bool(mask[0, 1]) is False

    # Node 2 attends to Node 0 and Node 1
    assert bool(mask[2, 0]) is True
    assert bool(mask[2, 1]) is True

    # Node 3 attends to Node 0
    assert bool(mask[3, 0]) is True
    # Crucial: Node 3 does NOT attend to Node 1 or Node 2 (sibling / branch isolation)
    assert bool(mask[3, 1]) is False
    assert bool(mask[3, 2]) is False

    # Node 2 does NOT attend to Node 3
    assert bool(mask[2, 3]) is False


def test_full_speculative_attention_mask():
    tree = SpeculativeTree()
    tree.add_node(token_id=10, depth=0)  # node 0
    tree.add_node(token_id=20, depth=1, parent_id=0)  # node 1

    prefix_len = 3
    full_mask = build_full_speculative_attention_mask(tree, prefix_len=prefix_len, as_bool=True)
    total_len = prefix_len + len(tree)  # 3 + 2 = 5
    assert full_mask.shape == (5, 5)

    # 1. Prefix causal
    assert bool(full_mask[0, 0]) is True
    assert bool(full_mask[0, 1]) is False
    assert bool(full_mask[2, 0]) is True
    assert bool(full_mask[2, 1]) is True

    # 2. Prefix never attends to tree
    for p in range(prefix_len):
        for t in range(prefix_len, total_len):
            assert bool(full_mask[p, t]) is False

    # 3. Tree always attends to full prefix
    for t in range(prefix_len, total_len):
        for p in range(prefix_len):
            assert bool(full_mask[t, p]) is True

    # 4. Tree-tree ancestor logic
    # Tree node 0 (idx 3) attends to itself
    assert bool(full_mask[3, 3]) is True
    # Tree node 1 (idx 4) attends to tree node 0 (idx 3)
    assert bool(full_mask[4, 3]) is True
    # Tree node 0 (idx 3) does NOT attend to tree node 1 (idx 4)
    assert bool(full_mask[3, 4]) is False


def test_float_attention_mask_values():
    tree = SpeculativeTree()
    tree.add_node(token_id=1, depth=0)
    tree.add_node(token_id=2, depth=1, parent_id=0)

    mask = build_tree_attention_mask(tree, as_bool=False)
    assert mask[0, 0].item() == 0.0
    assert mask[1, 0].item() == 0.0
    assert mask[0, 1].item() == float("-inf")
