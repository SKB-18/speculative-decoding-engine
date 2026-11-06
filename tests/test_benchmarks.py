"""
Tests for benchmark execution and scaling.
"""

import pytest
from benchmarks.run_benchmark import run_speculative_benchmark


def test_run_speculative_benchmark():
    results = run_speculative_benchmark(num_tokens=15)

    assert "autoregressive" in results
    assert "tree_speculative_medusa" in results
    assert "tree_speculative_ngram" in results

    ar = results["autoregressive"]
    medusa = results["tree_speculative_medusa"]

    assert ar["tokens_generated"] == 15
    assert medusa["tokens_generated"] == 15
    assert medusa["effective_throughput_multiplier"] >= 1.0
