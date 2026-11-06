"""
Tree Speculative Decoding Benchmark Suite.
Measures target forward pass reduction, acceptance rates, and effective speedup
comparing Tree-Attention Speculative Decoding against Standard Autoregressive Baseline.
"""

from typing import Dict, Any, List, Optional
import time
import json
import os
import torch
import torch.nn as nn

from tree.builder import TreeBuilder
from tree.node import SpeculativeTree
from drafter.medusa_head import MedusaDrafter
from drafter.ngram_tree import NGramTreeDrafter
from verifier.tree_verifier import TreeAttentionVerifier
from engine.speculative_engine import SpeculativeEngine


class SyntheticTargetModel(nn.Module):
    """
    Simulated Transformer Target Model with attention mask support.
    Produces deterministic token logits for benchmarking.
    """

    def __init__(self, vocab_size: int = 100, hidden_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        h = self.embedding(input_ids)
        if attention_mask is not None:
            m = attention_mask.squeeze()
            if m.ndim == 2:
                m_mean = torch.clamp(m, min=-10.0, max=0.0).mean(dim=-1, keepdim=True)  # [L, 1]
                h = h + m_mean * 0.001

        logits = self.proj(h)

        if output_hidden_states:
            return {"logits": logits, "hidden_states": [h]}
        return logits


def run_speculative_benchmark(num_tokens: int = 40) -> Dict[str, Any]:
    """
    Executes comparative benchmark across Autoregressive, Linear Draft, and Tree Speculative decoding.
    """
    torch.manual_seed(42)
    vocab_size = 120
    hidden_size = 64

    target_model = SyntheticTargetModel(vocab_size=vocab_size, hidden_size=hidden_size)
    prompt = [1, 5, 12, 18, 25, 33, 42, 50]

    # Pre-populate some repetitiveness in prompt for n-gram drafter
    repeated_pattern = [10, 20, 30, 40] * 3
    full_prompt = prompt + repeated_pattern

    results = {}

    # 1. Standard Autoregressive Baseline
    print("Running Autoregressive Baseline...")
    engine_base = SpeculativeEngine(
        target_model=target_model,
        drafter=NGramTreeDrafter(),
    )
    t0 = time.time()
    out_ar = engine_base.generate_autoregressive_baseline(full_prompt, max_new_tokens=num_tokens)
    ar_time = time.time() - t0
    ar_steps = out_ar.metrics["total_steps"]

    results["autoregressive"] = {
        "tokens_generated": len(out_ar.token_ids),
        "target_forward_passes": ar_steps,
        "elapsed_sec": round(ar_time, 4),
        "tokens_per_sec": round(len(out_ar.token_ids) / ar_time, 1),
        "speedup_multiplier": 1.0,
    }

    # 2. Tree-Structured Speculative Decoding (Medusa Drafter)
    print("Running Tree-Structured Speculative Decoding (Medusa Heads)...")
    medusa_drafter = MedusaDrafter(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=3,
        branching_factors=[2, 2, 2],
    )
    # Pre-train medusa head weights lightly to align with target model
    with torch.no_grad():
        for head in medusa_drafter.heads:
            head.lm_head.weight.copy_(target_model.proj.weight)

    engine_medusa = SpeculativeEngine(
        target_model=target_model,
        drafter=medusa_drafter,
        max_tree_nodes=14,
    )
    t0 = time.time()
    out_medusa = engine_medusa.generate(full_prompt, max_new_tokens=num_tokens)
    medusa_time = time.time() - t0

    medusa_metrics = engine_medusa.tracker.to_dict()
    results["tree_speculative_medusa"] = {
        "tokens_generated": len(out_medusa.token_ids),
        "target_forward_passes": medusa_metrics["total_steps"],
        "draft_tokens_evaluated": medusa_metrics["total_drafted"],
        "draft_tokens_accepted": medusa_metrics["total_accepted"],
        "acceptance_rate": medusa_metrics["acceptance_rate"],
        "mean_accepted_per_step": medusa_metrics["mean_accepted_per_step"],
        "forward_pass_reduction_pct": round(
            (1.0 - (medusa_metrics["total_steps"] / ar_steps)) * 100.0, 1
        ),
        "effective_throughput_multiplier": medusa_metrics["effective_throughput_ratio"],
        "elapsed_sec": round(medusa_time, 4),
        "tokens_per_sec": round(len(out_medusa.token_ids) / max(medusa_time, 1e-6), 1),
    }

    # 3. Dynamic N-Gram Tree Drafter
    print("Running Dynamic N-Gram Tree Drafter...")
    ngram_drafter = NGramTreeDrafter(min_ngram=2, max_branch_depth=4)
    engine_ngram = SpeculativeEngine(
        target_model=target_model,
        drafter=ngram_drafter,
        max_tree_nodes=12,
    )
    t0 = time.time()
    out_ngram = engine_ngram.generate(full_prompt, max_new_tokens=num_tokens)
    ngram_time = time.time() - t0

    ngram_metrics = engine_ngram.tracker.to_dict()
    results["tree_speculative_ngram"] = {
        "tokens_generated": len(out_ngram.token_ids),
        "target_forward_passes": ngram_metrics["total_steps"],
        "draft_tokens_evaluated": ngram_metrics["total_drafted"],
        "draft_tokens_accepted": ngram_metrics["total_accepted"],
        "acceptance_rate": ngram_metrics["acceptance_rate"],
        "mean_accepted_per_step": ngram_metrics["mean_accepted_per_step"],
        "effective_throughput_multiplier": ngram_metrics["effective_throughput_ratio"],
        "elapsed_sec": round(ngram_time, 4),
        "tokens_per_sec": round(len(out_ngram.token_ids) / max(ngram_time, 1e-6), 1),
    }

    # Save to disk
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark completed. Results written to: {out_path}")
    return results


if __name__ == "__main__":
    res = run_speculative_benchmark(num_tokens=40)
    print(json.dumps(res, indent=2))
