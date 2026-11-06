"""
Stress and Scaling Tests for Tree Speculative Decoding.
Evaluates deep branching topologies, large node counts, and batch scaling.
"""

import pytest
import torch
from tree.builder import TreeBuilder
from tree.mask import build_tree_attention_mask, build_full_speculative_attention_mask
from verifier.tree_verifier import TreeAttentionVerifier
from benchmarks.run_benchmark import SyntheticTargetModel


def test_large_tree_mask_scaling():
    """Verify tree mask generation scales efficiently to large candidate trees (e.g. 64 nodes)."""
    # 4 heads with branching factor 2 -> 2 + 4 + 8 + 16 + 32 = 62 nodes
    head_topk = [[(100 + d * 10 + i, 0.5) for i in range(2)] for d in range(5)]
    tree = TreeBuilder.from_medusa_heads(head_topk, branching_factors=[2, 2, 2, 2, 2])

    assert len(tree) >= 30
    mask = build_tree_attention_mask(tree, as_bool=True)
    assert mask.shape == (len(tree), len(tree))

    # Test causal property for all nodes: no node attends to a node with greater depth
    for i, node_i in tree.nodes.items():
        for j, node_j in tree.nodes.items():
            if node_j.depth > node_i.depth:
                assert not bool(mask[i, j])


def test_long_prefix_scaling():
    """Verify single-pass verification executes cleanly with S=256 prefix tokens."""
    target_model = SyntheticTargetModel(vocab_size=80, hidden_size=32)
    verifier = TreeAttentionVerifier(target_model=target_model, sampling_mode="greedy")

    prefix = torch.randint(0, 80, (256,), dtype=torch.long)
    paths = [[10, 20, 30], [10, 25, 35], [15, 25, 35]]
    tree = TreeBuilder.from_candidate_paths(paths)

    result = verifier.verify_step(prefix_tokens=prefix, tree=tree)
    assert len(result.total_emitted_tokens) >= 1
    assert isinstance(result.bonus_token_id, int)


def test_competing_branches_selection():
    """Verify greedy verifier selects the longest winning branch among multiple competing branches."""
    # Branch A: 10 -> 20 -> 30 (depth 2)
    # Branch B: 10 -> 25 (depth 1)
    # Branch C: 15 -> 40 (depth 1)
    paths = [[10, 20, 30], [10, 25], [15, 40]]
    tree = TreeBuilder.from_candidate_paths(paths)

    # Make target model predict Branch A (10 -> 20 -> 30)
    prefix_logits = torch.zeros(100)
    prefix_logits[10] = 10.0

    target_logits = torch.zeros(len(tree), 100)
    # Node with token 10 predicts 20
    node_10_id = [n.node_id for n in tree.nodes.values() if n.token_id == 10][0]
    target_logits[node_10_id, 20] = 10.0

    # Node with token 20 predicts 30
    node_20_id = [n.node_id for n in tree.nodes.values() if n.token_id == 20][0]
    target_logits[node_20_id, 30] = 10.0

    # Node with token 30 predicts bonus token 99
    node_30_id = [n.node_id for n in tree.nodes.values() if n.token_id == 30][0]
    target_logits[node_30_id, 99] = 10.0

    from verifier.sampler import GreedyTreeSampler
    result = GreedyTreeSampler.sample(tree, target_logits, prefix_logits)

    # Expect Branch A fully accepted (tokens 10, 20, 30) + bonus 99
    assert result.accepted_token_ids == [10, 20, 30]
    assert result.bonus_token_id == 99
    assert result.num_accepted == 3
