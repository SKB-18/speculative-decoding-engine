"""
Tree-Attention Speculative Verifier.
Executes a single parallel target model forward pass over an entire candidate tree
and computes accepted token branches via causal tree attention masking.
"""

from typing import Optional, Callable, Union, Literal, Any
import torch
import torch.nn as nn

from tree.node import SpeculativeTree
from tree.mask import build_full_speculative_attention_mask
from .sampler import VerificationResult, GreedyTreeSampler, StochasticSpeculativeSampler


class TreeAttentionVerifier:
    """
    Parallel verifier for tree-structured speculative candidate sequences.
    """

    def __init__(
        self,
        target_model: Union[nn.Module, Callable],
        sampling_mode: Literal["greedy", "stochastic"] = "greedy",
        temperature: float = 1.0,
    ):
        self.target_model = target_model
        self.sampling_mode = sampling_mode
        self.temperature = max(1e-4, temperature)

    def verify_step(
        self,
        prefix_tokens: torch.Tensor,
        tree: SpeculativeTree,
        past_key_values: Optional[Any] = None,
    ) -> VerificationResult:
        """
        Executes a single forward verification pass over prefix + speculative tree.

        Args:
            prefix_tokens: [1, S] or [S] context token IDs.
            tree: SpeculativeTree containing N candidate tokens.
            past_key_values: Optional KV cache state.

        Returns:
            VerificationResult containing accepted tokens and the bonus token.
        """
        # Ensure 2D tensor [1, S]
        if prefix_tokens.ndim == 1:
            prefix = prefix_tokens.unsqueeze(0)
        else:
            prefix = prefix_tokens

        device = prefix.device
        S = prefix.shape[-1]
        N = len(tree)

        if N == 0:
            # No speculative tokens: standard single-token forward
            with torch.no_grad():
                out = self._forward_model(prefix, mask=None, position_ids=None)
                logits = out["logits"] if isinstance(out, dict) else out
                last_logits = logits[0, -1, :]

            if self.sampling_mode == "greedy":
                bonus = int(torch.argmax(last_logits).item())
            else:
                probs = torch.softmax(last_logits / self.temperature, dim=-1)
                bonus = int(torch.multinomial(probs, num_samples=1).item())

            return VerificationResult(
                accepted_token_ids=[],
                accepted_node_ids=[],
                bonus_token_id=bonus,
            )

        # 1. Flatten candidate tokens and relative depths
        tree_tokens = tree.flatten_tokens(device=device).unsqueeze(0)  # [1, N]
        depth_offsets = tree.flatten_position_offsets(device=device)  # [N]

        # 2. Build full attention mask: [S + N, S + N]
        full_mask = build_full_speculative_attention_mask(
            tree=tree,
            prefix_len=S,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, S + N, S + N]

        # 3. Build position IDs: [1, S + N]
        prefix_positions = torch.arange(S, dtype=torch.long, device=device)
        tree_positions = S + depth_offsets
        position_ids = torch.cat([prefix_positions, tree_positions], dim=0).unsqueeze(0)

        # 4. Concatenate input IDs: [1, S + N]
        combined_input_ids = torch.cat([prefix, tree_tokens], dim=-1)

        # 5. Single parallel forward pass
        with torch.no_grad():
            out = self._forward_model(
                input_ids=combined_input_ids,
                mask=full_mask,
                position_ids=position_ids,
            )
            logits = out["logits"] if isinstance(out, dict) else out
            if logits.ndim == 4:
                logits = logits.squeeze(1)
            # logits: [1, S + N, V]

        # 6. Extract evaluation logits
        # Token at prefix index S-1 predicts root candidates (depth 0)
        prefix_last_logits = logits[0, S - 1, :]  # [V]
        # Candidate at index S + i predicts child candidates for node i
        target_logits_per_node = logits[0, S : S + N, :]  # [N, V]

        # 7. Sample accepted path
        if self.sampling_mode == "greedy":
            result = GreedyTreeSampler.sample(
                tree=tree,
                target_logits_per_node=target_logits_per_node,
                prefix_last_logits=prefix_last_logits,
            )
        else:
            result = StochasticSpeculativeSampler.sample(
                tree=tree,
                target_logits_per_node=target_logits_per_node,
                prefix_last_logits=prefix_last_logits,
                temperature=self.temperature,
            )

        return result

    def _forward_model(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Invokes target model with flexible signature support.
        """
        # Check callable signature or invoke directly
        try:
            return self.target_model(input_ids, attention_mask=mask, position_ids=position_ids)
        except TypeError:
            try:
                return self.target_model(input_ids, mask=mask)
            except TypeError:
                return self.target_model(input_ids)
