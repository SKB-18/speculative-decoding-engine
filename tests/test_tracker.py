"""
Unit tests for SpeculativeMetricsTracker.
"""

import pytest
from verifier.tracker import SpeculativeMetricsTracker


def test_tracker_computations():
    tracker = SpeculativeMetricsTracker()

    # Step 1: 4 drafted, 2 accepted
    tracker.record_step(num_drafted=4, num_accepted=2)
    # Step 2: 4 drafted, 1 accepted
    tracker.record_step(num_drafted=4, num_accepted=1)

    assert tracker.total_steps == 2
    assert tracker.total_drafted == 8
    assert tracker.total_accepted == 3
    assert tracker.total_bonus == 2
    assert tracker.total_emitted == 5  # 3 accepted + 2 bonus

    # acceptance rate: 3 / 8 = 0.375
    assert abs(tracker.acceptance_rate - 0.375) < 1e-4

    # mean accepted per step: 3 / 2 = 1.5
    assert abs(tracker.mean_accepted_per_step - 1.5) < 1e-4

    # throughput multiplier: 5 / 2 = 2.5x vs autoregressive
    assert abs(tracker.effective_throughput_ratio - 2.5) < 1e-4

    report = tracker.summary_report()
    assert "Throughput Multiplier" in report
    assert "2.50x" in report
