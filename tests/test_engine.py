"""
Unit tests for End-to-End SpeculativeEngine generation.
"""

import pytest
import torch
from engine.speculative_engine import SpeculativeEngine
from drafter.ngram_tree import NGramTreeDrafter
from benchmarks.run_benchmark import SyntheticTargetModel


def test_speculative_engine_generation_loop():
    target_model = SyntheticTargetModel(vocab_size=80, hidden_size=32)
    drafter = NGramTreeDrafter()
    engine = SpeculativeEngine(target_model=target_model, drafter=drafter)

    prompt = [1, 2, 3, 4, 1, 2, 3]
    out = engine.generate(prompt_tokens=prompt, max_new_tokens=15)

    assert len(out.token_ids) == 15
    assert out.tokens_per_sec > 0
    assert out.metrics["total_steps"] > 0


def test_speculative_engine_eos_termination():
    target_model = SyntheticTargetModel(vocab_size=80, hidden_size=32)
    drafter = NGramTreeDrafter()
    engine = SpeculativeEngine(target_model=target_model, drafter=drafter)

    # Force target model to emit token 50
    with torch.no_grad():
        target_model.proj.weight.fill_(0.0)
        target_model.proj.weight[50, :] = 10.0

    prompt = [1, 2, 3]
    out = engine.generate(prompt_tokens=prompt, max_new_tokens=20, eos_token_id=50)

    # Should stop early when 50 is emitted
    assert 50 in out.token_ids
    assert len(out.token_ids) <= 20
