"""
Unit tests for Medusa and N-Gram Speculative Drafters.
"""

import pytest
import torch
from drafter.medusa_head import MedusaHead, MedusaDrafter
from drafter.ngram_tree import NGramTreeDrafter


def test_medusa_head_shapes():
    hidden_size = 32
    vocab_size = 50
    head = MedusaHead(hidden_size=hidden_size, vocab_size=vocab_size)

    x = torch.randn(2, 4, hidden_size)
    logits = head(x)
    assert logits.shape == (2, 4, vocab_size)


def test_medusa_drafter_generate_tree():
    hidden_size = 32
    vocab_size = 50
    drafter = MedusaDrafter(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=3,
        branching_factors=[2, 2, 2],
    )

    h_last = torch.randn(hidden_size)
    context = torch.tensor([1, 2, 3])
    tree = drafter.generate_tree(context_tokens=context, hidden_state=h_last, max_nodes=12)

    assert len(tree) > 0
    assert len(tree) <= 12
    assert tree.get_max_depth() <= 3


def test_ngram_drafter_finds_pattern():
    drafter = NGramTreeDrafter(min_ngram=2, max_ngram=4, max_branch_depth=3)

    # Prompt has repeated sequence [10, 20, 30, 40]
    # Current suffix ends with [10, 20]
    tokens = [5, 6, 10, 20, 30, 40, 99, 100, 10, 20]
    tree = drafter.generate_tree(context_tokens=tokens, max_nodes=10)

    assert len(tree) > 0
    # Continuation should be 30, 40
    root = tree.nodes[tree.root_ids[0]]
    assert root.token_id == 30
    assert len(root.children_ids) > 0
    child = tree.nodes[root.children_ids[0]]
    assert child.token_id == 40


def test_ngram_drafter_empty_fallback():
    drafter = NGramTreeDrafter(min_ngram=3)
    # Short context with zero repetitions
    tokens = [1, 2]
    tree = drafter.generate_tree(context_tokens=tokens)
    assert len(tree) == 0
