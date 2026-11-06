"""
Speculative Decoding Telemetry and Speedup Metrics Tracker.
Monitors acceptance rates, effective tokens per target pass, and latency speedup.
"""

from typing import Dict, Any, List
import time


class SpeculativeMetricsTracker:
    """
    Tracks step-by-step speculative decoding statistics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time: float = time.time()
        self.total_steps: int = 0
        self.total_drafted: int = 0
        self.total_accepted: int = 0
        self.total_bonus: int = 0
        self.step_history: List[Dict[str, Any]] = []

    def record_step(
        self,
        num_drafted: int,
        num_accepted: int,
        step_latency_ms: float = 0.0,
    ):
        """
        Records the outcome of a single speculative verification step.
        """
        self.total_steps += 1
        self.total_drafted += num_drafted
        self.total_accepted += num_accepted
        self.total_bonus += 1

        self.step_history.append({
            "step": self.total_steps,
            "drafted": num_drafted,
            "accepted": num_accepted,
            "emitted": num_accepted + 1,
            "latency_ms": step_latency_ms,
        })

    @property
    def total_emitted(self) -> int:
        """Total tokens produced = accepted speculative tokens + 1 bonus token per step."""
        return self.total_accepted + self.total_bonus

    @property
    def acceptance_rate(self) -> float:
        """Proportion of speculative draft tokens accepted."""
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    @property
    def mean_accepted_per_step(self) -> float:
        """Average speculative tokens accepted per target forward pass."""
        if self.total_steps == 0:
            return 0.0
        return self.total_accepted / self.total_steps

    @property
    def effective_throughput_ratio(self) -> float:
        """
        Tokens generated per target model forward pass.
        In standard autoregressive decoding, this ratio is exactly 1.0.
        In speculative decoding, this ratio is (1 + mean_accepted).
        """
        if self.total_steps == 0:
            return 1.0
        return self.total_emitted / self.total_steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_drafted": self.total_drafted,
            "total_accepted": self.total_accepted,
            "total_emitted": self.total_emitted,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "mean_accepted_per_step": round(self.mean_accepted_per_step, 4),
            "effective_throughput_ratio": round(self.effective_throughput_ratio, 4),
        }

    def summary_report(self) -> str:
        d = self.to_dict()
        return (
            f"Speculative Decoding Telemetry:\n"
            f"  - Target Forward Passes : {d['total_steps']}\n"
            f"  - Total Tokens Emitted  : {d['total_emitted']}\n"
            f"  - Draft Tokens Evaluated: {d['total_drafted']}\n"
            f"  - Draft Tokens Accepted : {d['total_accepted']}\n"
            f"  - Acceptance Rate (alpha): {d['acceptance_rate'] * 100:.1f}%\n"
            f"  - Mean Accepted / Step  : {d['mean_accepted_per_step']:.2f} tokens\n"
            f"  - Throughput Multiplier : {d['effective_throughput_ratio']:.2f}x vs Autoregressive\n"
        )
