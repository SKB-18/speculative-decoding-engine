"""
Unit tests for TreeAttentionVerifier parallel execution.
"""

import pytest
import torch
import torch.nn as nn
from tree.builder import TreeBuilder
from verifier.tree_verifier import TreeAttentionVerifier
from benchmarks.run_benchmark import SyntheticTargetModel


def test_tree_verifier_single_pass_execution():
    target_model = SyntheticTargetModel(vocab_size=60, hidden_size=32)
    verifier = TreeAttentionVerifier(target_model=target_model, sampling_mode="greedy")

    prefix = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    paths = [[10, 20], [10, 30]]
    tree = TreeBuilder.from_candidate_paths(paths)

    result = verifier.verify_step(prefix_tokens=prefix, tree=tree)
    assert isinstance(result.accepted_token_ids, list)
    assert isinstance(result.bonus_token_id, int)
    assert len(result.total_emitted_tokens) >= 1


def test_tree_verifier_zero_tree_fallback():
    target_model = SyntheticTargetModel(vocab_size=40, hidden_size=16)
    verifier = TreeAttentionVerifier(target_model=target_model, sampling_mode="greedy")

    prefix = torch.tensor([5, 6, 7])
    from tree.node import SpeculativeTree
    empty_tree = SpeculativeTree()

    result = verifier.verify_step(prefix_tokens=prefix, tree=empty_tree)
    assert result.accepted_token_ids == []
    assert isinstance(result.bonus_token_id, int)
