"""
Unit tests for Greedy and Stochastic Speculative Acceptance Samplers.
"""

import pytest
import torch
from tree.node import SpeculativeTree
from verifier.sampler import GreedyTreeSampler, StochasticSpeculativeSampler, VerificationResult


def test_greedy_sampler_exact_path_acceptance():
    # Tree:
    # 0: token 10
    # ├── 1: token 20
    # └── 2: token 25
    tree = SpeculativeTree()
    tree.add_node(token_id=10, depth=0)  # 0
    tree.add_node(token_id=20, depth=1, parent_id=0)  # 1
    tree.add_node(token_id=25, depth=1, parent_id=0)  # 2

    # Target model predictions:
    # At prefix end: predicts 10 (accepts node 0)
    # At node 0: predicts 20 (accepts node 1)
    # At node 1: predicts 99 (bonus token!)
    # At node 2: predicts 88
    prefix_logits = torch.zeros(100)
    prefix_logits[10] = 10.0

    target_logits_per_node = torch.zeros(3, 100)
    target_logits_per_node[0, 20] = 10.0  # node 0 predicts 20 -> matches node 1
    target_logits_per_node[1, 99] = 10.0  # node 1 predicts 99
    target_logits_per_node[2, 88] = 10.0

    result: VerificationResult = GreedyTreeSampler.sample(
        tree=tree,
        target_logits_per_node=target_logits_per_node,
        prefix_last_logits=prefix_logits,
    )

    assert result.accepted_token_ids == [10, 20]
    assert result.accepted_node_ids == [0, 1]
    assert result.bonus_token_id == 99
    assert result.total_emitted_tokens == [10, 20, 99]
    assert result.num_accepted == 2


def test_greedy_sampler_root_rejection():
    tree = SpeculativeTree()
    tree.add_node(token_id=10, depth=0)
    tree.add_node(token_id=20, depth=1, parent_id=0)

    # Prefix predicts 77 (does NOT match root candidate 10)
    prefix_logits = torch.zeros(100)
    prefix_logits[77] = 10.0

    target_logits_per_node = torch.zeros(2, 100)

    result = GreedyTreeSampler.sample(
        tree=tree,
        target_logits_per_node=target_logits_per_node,
        prefix_last_logits=prefix_logits,
    )

    # 0 tokens accepted, bonus token is 77
    assert result.accepted_token_ids == []
    assert result.bonus_token_id == 77
    assert result.total_emitted_tokens == [77]


def test_greedy_sampler_empty_tree():
    tree = SpeculativeTree()
    prefix_logits = torch.zeros(50)
    prefix_logits[42] = 5.0
    target_logits = torch.empty(0, 50)

    result = GreedyTreeSampler.sample(tree, target_logits, prefix_logits)
    assert result.accepted_token_ids == []
    assert result.bonus_token_id == 42


def test_stochastic_sampler_runs():
    tree = SpeculativeTree()
    tree.add_node(token_id=5, depth=0, draft_prob=0.8)

    prefix_logits = torch.randn(50)
    target_logits = torch.randn(1, 50)

    result = StochasticSpeculativeSampler.sample(
        tree=tree,
        target_logits_per_node=target_logits,
        prefix_last_logits=prefix_logits,
        temperature=1.0,
    )

    assert isinstance(result.bonus_token_id, int)
    assert 0 <= result.bonus_token_id < 50
