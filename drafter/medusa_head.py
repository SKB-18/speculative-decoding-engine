"""
Medusa-Style Multi-Head Speculative Drafter.
Generates multi-step branching candidate trees directly from the target model's
final hidden representations using lightweight parallel residual classification heads.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDrafter
from tree.node import SpeculativeTree
from tree.builder import TreeBuilder


class MedusaHead(nn.Module):
    """
    A single Medusa residual prediction head.
    Predicts logits for token offset (t + k + 1) given hidden state at token t.
    Structure:
      x -> Linear(d, d) -> SiLU -> Linear(d, d) + x -> Linear(d, vocab_size)
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)  # Zero init for residual identity at step 0
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        hidden_state: [..., hidden_size]
        Returns: [..., vocab_size] logits
        """
        res = hidden_state
        h = self.fc2(self.act(self.fc1(hidden_state))) + res
        return self.lm_head(h)


class MedusaDrafter(nn.Module, BaseDrafter):
    """
    Medusa Multi-Head Drafter containing K parallel heads.
    Predicts K offsets into the future simultaneously.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 3,
        branching_factors: Optional[List[int]] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        # Default Medusa branching topology (e.g. [2, 3, 2] = 2 + 2*3 + 2*3*2 = 20 max nodes)
        self.branching_factors = branching_factors or ([2] * num_heads)
        self.temperature = max(1e-4, temperature)

        self.heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size) for _ in range(num_heads)
        ])

    def forward(self, hidden_state: torch.Tensor) -> List[torch.Tensor]:
        """
        hidden_state: [..., hidden_size] (typically the last token's representation)
        Returns: List of [..., vocab_size] logits for each head 0..K-1
        """
        return [head(hidden_state) for head in self.heads]

    def generate_tree(
        self,
        context_tokens: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        max_nodes: int = 16,
    ) -> SpeculativeTree:
        """
        Predicts top candidates from each head and builds a SpeculativeTree.
        """
        if hidden_state is None:
            raise ValueError("MedusaDrafter requires hidden_state to predict candidate tokens.")

        # Ensure hidden_state is for the final token: [D]
        if hidden_state.ndim == 3:
            h_last = hidden_state[:, -1, :].squeeze(0)
        elif hidden_state.ndim == 2:
            h_last = hidden_state[-1, :]
        else:
            h_last = hidden_state

        device = h_last.device

        with torch.no_grad():
            all_logits = [head(h_last) for head in self.heads]  # List of [vocab_size]

            head_topk: List[List[Tuple[int, float]]] = []
            for k, logits in enumerate(all_logits):
                b_factor = self.branching_factors[min(k, len(self.branching_factors) - 1)]
                probs = F.softmax(logits / self.temperature, dim=-1)
                top_probs, top_indices = torch.topk(probs, k=min(b_factor, self.vocab_size))

                candidates = [
                    (int(top_indices[i].item()), float(top_probs[i].item()))
                    for i in range(len(top_indices))
                ]
                head_topk.append(candidates)

        tree = TreeBuilder.from_medusa_heads(
            head_topk=head_topk,
            branching_factors=self.branching_factors,
            max_nodes=max_nodes,
        )
        return tree
