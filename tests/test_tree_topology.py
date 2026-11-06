"""
Unit tests for Speculative Tree Data Structures and Topology.
"""

import pytest
import torch
from tree.node import SpeculativeTree, SpeculativeTreeNode


def test_add_node_parent_child_links():
    tree = SpeculativeTree()
    root = tree.add_node(token_id=101, depth=0, parent_id=None, draft_prob=0.9)
    assert root.node_id == 0
    assert len(tree.root_ids) == 1
    assert tree.root_ids[0] == 0

    child = tree.add_node(token_id=202, depth=1, parent_id=0, draft_prob=0.8)
    assert child.node_id == 1
    assert child.parent_id == 0
    assert 1 in root.children_ids
    assert len(tree) == 2


def test_ancestors_computation():
    tree = SpeculativeTree()
    # 0 -> 1 -> 2
    # 0 -> 3
    n0 = tree.add_node(token_id=10, depth=0)
    n1 = tree.add_node(token_id=20, depth=1, parent_id=0)
    n2 = tree.add_node(token_id=30, depth=2, parent_id=1)
    n3 = tree.add_node(token_id=40, depth=1, parent_id=0)

    # Node 2 ancestors
    anc2 = tree.get_ancestors(2, include_self=True)
    assert anc2 == {0, 1, 2}

    anc2_no_self = tree.get_ancestors(2, include_self=False)
    assert anc2_no_self == {0, 1}

    # Node 3 ancestors
    anc3 = tree.get_ancestors(3, include_self=True)
    assert anc3 == {0, 3}
    assert 1 not in anc3
    assert 2 not in anc3


def test_path_from_root():
    tree = SpeculativeTree()
    tree.add_node(token_id=100, depth=0)  # 0
    tree.add_node(token_id=200, depth=1, parent_id=0)  # 1
    tree.add_node(token_id=300, depth=2, parent_id=1)  # 2

    path = tree.get_path_from_root(2)
    assert path == [0, 1, 2]

    tokens = tree.get_token_sequence(2)
    assert tokens == [100, 200, 300]


def test_flatten_tokens_and_offsets():
    tree = SpeculativeTree()
    tree.add_node(token_id=55, depth=0)
    tree.add_node(token_id=66, depth=1, parent_id=0)
    tree.add_node(token_id=77, depth=1, parent_id=0)

    tokens = tree.flatten_tokens()
    assert torch.equal(tokens, torch.tensor([55, 66, 77]))

    offsets = tree.flatten_position_offsets()
    assert torch.equal(offsets, torch.tensor([0, 1, 1]))


def test_invalid_parent_raises():
    tree = SpeculativeTree()
    with pytest.raises(ValueError, match="Parent node 999 does not exist"):
        tree.add_node(token_id=10, depth=1, parent_id=999)


def test_ascii_tree_visualization():
    tree = SpeculativeTree()
    tree.add_node(token_id=1, depth=0)
    tree.add_node(token_id=2, depth=1, parent_id=0)
    rep = tree.to_ascii_tree()
    assert "Node 0" in rep
    assert "Node 1" in rep
