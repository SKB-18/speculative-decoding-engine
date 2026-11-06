"""
Mathematical Distribution-Preservation Tests for Stochastic Speculative Sampling.
Verifies that speculative decoding under stochastic residual distribution matching
preserves the exact target model probability distribution (Zero Perplexity Distortion).
"""

import pytest
import torch
import torch.nn.functional as F
from tree.node import SpeculativeTree
from verifier.sampler import StochasticSpeculativeSampler


def test_stochastic_sampling_exact_distribution_matching():
    """
    Empirically verifies that StochasticSpeculativeSampler produces token frequencies
    statistically indistinguishable from the target distribution P(x) (TVD < 0.08).
    """
    torch.manual_seed(42)
    vocab_size = 5
    # True target distribution P(x)
    target_logits = torch.tensor([2.0, 1.0, 0.5, 0.1, -1.0])
    target_probs = F.softmax(target_logits, dim=-1)

    # Imperfect drafter proposing token 0 with 80% prob, token 1 with 20%
    num_trials = 1200
    sampled_tokens = []

    for _ in range(num_trials):
        tree = SpeculativeTree()
        # Draft proposal
        tree.add_node(token_id=0, depth=0, draft_prob=0.8)
        tree.add_node(token_id=1, depth=0, draft_prob=0.2)

        # Node predictions (not reached if trial checks roots)
        node_logits = torch.randn(2, vocab_size)

        result = StochasticSpeculativeSampler.sample(
            tree=tree,
            target_logits_per_node=node_logits,
            prefix_last_logits=target_logits,
            temperature=1.0,
        )

        # The first token emitted is either the accepted root or the residual bonus token
        first_token = result.total_emitted_tokens[0]
        sampled_tokens.append(first_token)

    # Compute empirical distribution
    counts = torch.bincount(torch.tensor(sampled_tokens), minlength=vocab_size).float()
    empirical_probs = counts / num_trials

    # Compute Total Variation Distance (TVD)
    tvd = 0.5 * torch.sum(torch.abs(empirical_probs - target_probs)).item()

    # TVD should be small (< 0.08 with 1200 trials)
    assert tvd < 0.08, f"Distribution diverged: TVD={tvd:.4f}, expected < 0.08"
