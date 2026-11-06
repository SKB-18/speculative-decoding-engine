"""
Edge Case and Boundary Condition Tests for Tree Speculative Decoding.
"""

import pytest
import torch
from tree.builder import TreeBuilder
from tree.node import SpeculativeTree
from verifier.tree_verifier import TreeAttentionVerifier
from verifier.sampler import GreedyTreeSampler
from benchmarks.run_benchmark import SyntheticTargetModel


def test_single_token_prefix():
    """Verify single-pass verification functions properly with a 1-token prompt (S=1)."""
    target_model = SyntheticTargetModel(vocab_size=40, hidden_size=16)
    verifier = TreeAttentionVerifier(target_model=target_model, sampling_mode="greedy")

    prefix = torch.tensor([5])
    paths = [[10, 20]]
    tree = TreeBuilder.from_candidate_paths(paths)

    result = verifier.verify_step(prefix_tokens=prefix, tree=tree)
    assert len(result.total_emitted_tokens) >= 1
    assert isinstance(result.bonus_token_id, int)


def test_high_vocab_token_indices():
    """Verify speculative tree handles large token IDs without integer truncation."""
    tree = SpeculativeTree()
    tree.add_node(token_id=151643, depth=0)  # Qwen/Llama large vocabulary index
    tree.add_node(token_id=151644, depth=1, parent_id=0)

    tokens = tree.flatten_tokens()
    assert tokens[0].item() == 151643
    assert tokens[1].item() == 151644


def test_complete_tree_acceptance():
    """Verify when all nodes on a branch are accepted, the bonus token is sampled from the leaf node."""
    tree = SpeculativeTree()
    tree.add_node(token_id=1, depth=0)  # node 0
    tree.add_node(token_id=2, depth=1, parent_id=0)  # node 1
    tree.add_node(token_id=3, depth=2, parent_id=1)  # node 2

    # Target predictions match exactly
    prefix_logits = torch.zeros(10)
    prefix_logits[1] = 10.0

    target_logits = torch.zeros(3, 10)
    target_logits[0, 2] = 10.0
    target_logits[1, 3] = 10.0
    target_logits[2, 7] = 10.0  # Leaf predicts 7

    result = GreedyTreeSampler.sample(tree, target_logits, prefix_logits)
    assert result.accepted_token_ids == [1, 2, 3]
    assert result.bonus_token_id == 7
    assert result.num_accepted == 3
    assert result.total_emitted_tokens == [1, 2, 3, 7]
