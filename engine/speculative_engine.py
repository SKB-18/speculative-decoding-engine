"""
End-to-End Tree Speculative Decoding Engine.
Coordinates the target LLM, speculative tree drafter, tree attention verifier,
and metrics tracker for high-throughput acceleration.
"""

from typing import List, Optional, Union, Dict, Any, Callable
import time
import torch
import torch.nn as nn

from tree.node import SpeculativeTree
from drafter.base import BaseDrafter
from verifier.tree_verifier import TreeAttentionVerifier
from verifier.tracker import SpeculativeMetricsTracker
from verifier.sampler import VerificationResult


class GenerationOutput:
    """
    Result container for a speculative generation run.
    """

    def __init__(
        self,
        token_ids: List[int],
        metrics: Dict[str, Any],
        elapsed_time_sec: float,
        tokens_per_sec: float,
        speculative_speedup: float = 1.0,
    ):
        self.token_ids = token_ids
        self.metrics = metrics
        self.elapsed_time_sec = elapsed_time_sec
        self.tokens_per_sec = tokens_per_sec
        self.speculative_speedup = speculative_speedup

    def __repr__(self) -> str:
        return (
            f"GenerationOutput(tokens={len(self.token_ids)}, "
            f"speed={self.tokens_per_sec:.1f} tok/s, "
            f"multiplier={self.speculative_speedup:.2f}x, "
            f"time={self.elapsed_time_sec:.3f}s)"
        )


class SpeculativeEngine:
    """
    High-level orchestrator for tree-structured speculative decoding.
    """

    def __init__(
        self,
        target_model: Union[nn.Module, Callable],
        drafter: BaseDrafter,
        sampling_mode: str = "greedy",
        temperature: float = 1.0,
        max_tree_nodes: int = 16,
    ):
        self.target_model = target_model
        self.drafter = drafter
        self.max_tree_nodes = max_tree_nodes
        self.verifier = TreeAttentionVerifier(
            target_model=target_model,
            sampling_mode=sampling_mode,
            temperature=temperature,
        )
        self.tracker = SpeculativeMetricsTracker()

    def generate(
        self,
        prompt_tokens: Union[torch.Tensor, List[int]],
        max_new_tokens: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> GenerationOutput:
        """
        Executes tree speculative decoding to generate up to max_new_tokens.
        """
        if isinstance(prompt_tokens, list):
            context = torch.tensor([prompt_tokens], dtype=torch.long)
        elif prompt_tokens.ndim == 1:
            context = prompt_tokens.unsqueeze(0)
        else:
            context = prompt_tokens.clone()

        device = context.device
        self.tracker.reset()
        generated_tokens: List[int] = []
        t0 = time.time()

        hidden_state: Optional[torch.Tensor] = None

        while len(generated_tokens) < max_new_tokens:
            step_t0 = time.time()

            # 1. Obtain hidden state if needed by the drafter
            if getattr(self.drafter, "heads", None) is not None:
                # Medusa drafter needs hidden state from target model
                with torch.no_grad():
                    # Evaluate context to get last hidden state
                    out = self._get_model_hidden(context)
                    hidden_state = out

            # 2. Drafter constructs candidate tree
            tree = self.drafter.generate_tree(
                context_tokens=context,
                hidden_state=hidden_state,
                max_nodes=self.max_tree_nodes,
            )

            # 3. Verifier tests entire candidate tree in a single parallel pass
            result: VerificationResult = self.verifier.verify_step(
                prefix_tokens=context,
                tree=tree,
            )

            step_latency_ms = (time.time() - step_t0) * 1000.0

            # 4. Extract emitted tokens
            emitted = result.total_emitted_tokens
            if len(generated_tokens) + len(emitted) > max_new_tokens:
                emitted = emitted[: max_new_tokens - len(generated_tokens)]

            generated_tokens.extend(emitted)

            # 5. Record step metrics
            self.tracker.record_step(
                num_drafted=len(tree),
                num_accepted=result.num_accepted,
                step_latency_ms=step_latency_ms,
            )

            # 6. Update context
            new_tokens_tensor = torch.tensor([emitted], dtype=torch.long, device=device)
            context = torch.cat([context, new_tokens_tensor], dim=-1)

            # 7. Check for EOS token
            if eos_token_id is not None and eos_token_id in emitted:
                break

        elapsed = time.time() - t0
        tps = len(generated_tokens) / max(elapsed, 1e-6)
        speedup = self.tracker.effective_throughput_ratio

        return GenerationOutput(
            token_ids=generated_tokens,
            metrics=self.tracker.to_dict(),
            elapsed_time_sec=elapsed,
            tokens_per_sec=tps,
            speculative_speedup=speedup,
        )

    def generate_autoregressive_baseline(
        self,
        prompt_tokens: Union[torch.Tensor, List[int]],
        max_new_tokens: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> GenerationOutput:
        """
        Executes standard single-token autoregressive decoding as ground truth baseline.
        """
        if isinstance(prompt_tokens, list):
            context = torch.tensor([prompt_tokens], dtype=torch.long)
        elif prompt_tokens.ndim == 1:
            context = prompt_tokens.unsqueeze(0)
        else:
            context = prompt_tokens.clone()

        device = context.device
        generated_tokens: List[int] = []
        t0 = time.time()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.target_model(context)
                logits = out["logits"] if isinstance(out, dict) else out
                next_token = int(torch.argmax(logits[0, -1, :]).item())

            generated_tokens.append(next_token)
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            context = torch.cat([context, next_tensor], dim=-1)

            if eos_token_id is not None and next_token == eos_token_id:
                break

        elapsed = time.time() - t0
        tps = len(generated_tokens) / max(elapsed, 1e-6)

        return GenerationOutput(
            token_ids=generated_tokens,
            metrics={"total_steps": len(generated_tokens)},
            elapsed_time_sec=elapsed,
            tokens_per_sec=tps,
            speculative_speedup=1.0,
        )

    def _get_model_hidden(self, context: torch.Tensor) -> torch.Tensor:
        """Helper to extract the last hidden representation from target model."""
        try:
            out = self.target_model(context, output_hidden_states=True)
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                return out.hidden_states[-1][:, -1, :]
            elif isinstance(out, dict) and "hidden_states" in out:
                return out["hidden_states"][-1][:, -1, :]
        except Exception:
            pass
        # Fallback dummy hidden vector
        return torch.randn(context.shape[0], getattr(self.drafter, "hidden_size", 64), device=context.device)
