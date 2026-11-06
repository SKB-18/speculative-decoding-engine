"""
Unit tests for TreeBuilder factory and pruning.
"""

import pytest
from tree.node import SpeculativeTree
from tree.builder import TreeBuilder


def test_builder_from_candidate_paths():
    paths = [
        [10, 20, 30],
        [10, 20, 40],
        [10, 50],
    ]
    tree = TreeBuilder.from_candidate_paths(paths)

    # Common prefix [10, 20] should be merged:
    # 0: 10
    # ├── 1: 20
    # │   ├── 2: 30
    # │   └── 3: 40
    # └── 4: 50
    # Total nodes: 5 (instead of 8 in unmerged paths)
    assert len(tree) == 5
    assert len(tree.root_ids) == 1
    root = tree.nodes[tree.root_ids[0]]
    assert root.token_id == 10
    assert root.depth == 0

    # Node 1 is child of root
    node1 = tree.nodes[1]
    assert node1.token_id == 20
    assert node1.parent_id == 0
    assert 2 in node1.children_ids
    assert 3 in node1.children_ids


def test_builder_from_medusa_heads():
    # 3 heads
    head_topk = [
        [(10, 0.9), (11, 0.1)],           # head 0: 2 candidates
        [(20, 0.8), (21, 0.2)],           # head 1: 2 candidates
        [(30, 0.9)],                      # head 2: 1 candidate
    ]
    branching = [2, 2, 1]
    tree = TreeBuilder.from_medusa_heads(head_topk, branching_factors=branching)

    # Level 0: 2 roots (nodes 0, 1)
    # Level 1: 2 parents * 2 children = 4 nodes (2, 3, 4, 5)
    # Level 2: 4 parents * 1 child = 4 nodes (6, 7, 8, 9)
    # Total = 2 + 4 + 4 = 10 nodes
    assert len(tree) == 10
    assert len(tree.root_ids) == 2
    assert tree.get_max_depth() == 3


def test_prune_tree_budget():
    paths = [
        [1, 2, 3, 4],
        [1, 5, 6, 7],
    ]
    tree = TreeBuilder.from_candidate_paths(paths)
    assert len(tree) == 7

    # Prune to max 4 nodes
    pruned = TreeBuilder.prune_tree(tree, max_nodes=4)
    assert len(pruned) <= 4

    # Every node in pruned tree must have its parent present
    for nid, node in pruned.nodes.items():
        if node.parent_id is not None:
            assert node.parent_id in pruned.nodes
