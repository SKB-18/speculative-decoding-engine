# Tree-Structured Speculative Decoding Engine

[![Tests](https://img.shields.io/badge/tests-38%2F38%20passed-brightgreen.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](#)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](#)

A high-performance, first-principles implementation of **Tree-Structured Speculative Decoding** (Medusa/EAGLE style) with causal tree-attention masks, multi-candidate parallel verification in a single forward pass, greedy and distribution-preserving stochastic samplers, and empirical throughput benchmarks in PyTorch.

---

## Architecture Overview

Traditional autoregressive decoding requires $K$ sequential forward passes through a Large Language Model to generate $K$ tokens. Speculative decoding amortizes memory bandwidth by drafting multiple candidate tokens with lightweight drafters and verifying them in parallel with the target model.

Tree-structured speculative decoding extends linear drafting to a **prefix trie (tree)** of multiple candidate paths. In a single forward pass, the target model verifies an entire candidate tree by constructing a **2D/3D causal tree-attention mask** where each candidate token attends strictly to its ancestors and the prompt prefix, without cross-branch causal leakage.

```
                      Prompt Prefix Tokens [P_0, P_1, ..., P_L]
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
               Branch A (Depth 1)                Branch B (Depth 1)
                  Token: 101                        Token: 204
                 (Offset +1)                       (Offset +1)
                      │                                 │
             ┌────────┴────────┐                        │
             ▼                 ▼                        ▼
       Node A1 (Depth 2)  Node A2 (Depth 2)       Node B1 (Depth 2)
          Token: 305         Token: 408              Token: 509
         (Offset +2)        (Offset +2)             (Offset +2)
             │
             ▼
       Node A1a (Depth 3)
          Token: 612
         (Offset +3)
```

### Flattened Token Buffer & Causal Tree Mask

The tree nodes are flattened via breadth-first search into a 1D tensor of length $N$. A causal attention mask $M \in \mathbb{R}^{N \times N}$ is constructed such that:

$$M_{i, j} = \begin{cases} 0 & \text{if } j \in \text{Ancestors}(i) \cup \{i\} \\ -\infty & \text{otherwise} \end{cases}$$

Combined with the prompt prefix of length $L$, the full speculative attention mask $M_{\text{full}} \in \mathbb{R}^{(L+N) \times (L+N)}$ enforces:
1. **Prompt tokens** attend causally only to preceding prompt tokens.
2. **Tree candidate tokens** attend fully to all $L$ prompt tokens.
3. **Tree candidate tokens** attend to preceding tree tokens if and only if they are ancestors in the candidate trie.

This invariant guarantees that hidden states computed for candidate $i$ are numerically identical to those computed if candidate $i$'s branch were processed in isolation.

---

## Mathematical Foundations & Sampling Algorithms

### 1. Single Forward Pass Multi-Candidate Verification

Given prompt hidden state $h_L$ and flattened candidate tokens $[t_1, \dots, t_N]$, the target LLM performs a single forward pass with $M_{\text{full}}$:
$$[\mathbf{z}_1, \dots, \mathbf{z}_N] = \text{TargetModel}([P_1, \dots, P_L, t_1, \dots, t_N], M_{\text{full}})_{L:L+N}$$
where $\mathbf{z}_i$ represents the target model's output logits predicting the successor token of candidate node $i$.

### 2. Longest-Prefix Greedy Acceptance

In greedy decoding ($T=0$):
1. For each branch starting from root children, candidate node $i$ with token $t_i$ is accepted if:
   $$t_i = \arg\max_v z_{\text{parent}(i)}[v]$$
2. Traverse down the tree greedily; the branch with the maximum depth of consecutive matches is selected.
3. **Bonus Token**: The target model's logits at the deepest accepted node $i_{\text{last}}$ yield a guaranteed extra token:
   $$t_{\text{bonus}} = \arg\max_v z_{i_{\text{last}}}[v]$$
   guaranteeing progress of at least $1$ token per step even on zero candidate acceptances.

### 3. Distribution-Preserving Stochastic Sampling (Leviathan / SpecInfer)

For temperature $T > 0$, stochastic speculative sampling guarantees that the output token distribution exactly matches the target model distribution $P(x)$, resulting in **Zero Perplexity Degradation**:

1. Let drafter candidate path have draft probabilities $Q(c)$ and target probabilities $P(c)$.
2. Sample candidate $c \sim Q(c)$.
3. Accept $c$ with probability:
   $$\alpha = \min\left(1, \frac{P(c)}{Q(c)}\right)$$
4. If accepted, recurse down children nodes.
5. If rejected, stop branch traversal and resample a bonus token from the normalized positive residual distribution:
   $$x_{\text{resample}} \sim \frac{\max(0, P(x) - Q(x))}{\sum_y \max(0, P(y) - Q(y))}$$

**Proof of Exact Equivalence**:
$$P_{\text{emit}}(x) = Q(x) \min\left(1, \frac{P(x)}{Q(x)}\right) + (1 - \alpha) \frac{\max(0, P(x) - Q(x))}{\sum_y \max(0, P(y) - Q(y))} \equiv P(x)$$

---

## Directory Structure

```
speculative-decoding-engine/
├── tree/
│   ├── __init__.py
│   ├── node.py           # SpeculativeTreeNode & SpeculativeTree topology
│   ├── mask.py           # 2D/3D causal tree-attention masks (bool & additive -inf)
│   └── builder.py        # Prefix trie builder from candidate paths & Medusa heads
├── drafter/
│   ├── __init__.py
│   ├── base.py           # BaseDrafter abstract interface
│   ├── medusa_head.py    # Multi-head residual MLP drafter (Medusa style)
│   └── ngram_tree.py     # Prompt-context n-gram continuation drafter
├── verifier/
│   ├── __init__.py
│   ├── sampler.py        # GreedyTreeSampler & StochasticSpeculativeSampler
│   ├── tree_verifier.py  # Parallel single-pass TreeAttentionVerifier
│   └── tracker.py        # Metrics: acceptance rate, mean accepted tokens, speedup
├── engine/
│   ├── __init__.py
│   └── speculative_engine.py  # Complete end-to-end SpeculativeEngine pipeline
├── benchmarks/
│   ├── __init__.py
│   └── run_benchmark.py  # Synthetic & empirical throughput benchmarking suite
├── tests/
│   ├── test_tree_topology.py
│   ├── test_tree_mask.py
│   ├── test_tree_builder.py
│   ├── test_drafters.py
│   ├── test_sampler.py
│   ├── test_verifier.py
│   ├── test_engine.py
│   ├── test_tracker.py
│   ├── test_cli.py
│   ├── test_benchmarks.py
│   ├── test_stress.py
│   ├── test_distribution_preservation.py
│   └── test_edge_cases.py
├── cli.py                # Command-line interface
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quickstart & CLI Usage

### 1. Installation

```bash
git clone https://github.com/SKB-18/speculative-decoding-engine.git
cd speculative-decoding-engine
pip install -e .
```

### 2. Inspect Drafter Architecture

```bash
python cli.py inspect
```

### 3. Visualize Speculative Trees

```bash
python cli.py tree --num-candidates 3 --depth 3
```
Output:
```
Speculative Trie Topology (Nodes: 7, Max Depth: 3)
└── [1] token=100 (depth=1)
    ├── [2] token=200 (depth=2)
    │   └── [4] token=400 (depth=3)
    └── [3] token=300 (depth=2)
└── [5] token=101 (depth=1)
    └── [6] token=201 (depth=2)
```

### 4. Run Single-Step Parallel Verification

```bash
python cli.py verify --prefix-len 16 --tree-nodes 8 --sampler greedy
```

### 5. Run Speculative Decoding Benchmarks

```bash
python cli.py benchmark --num-steps 50 --tree-size 8
```

Output:
```
================================================================================
SPECULATIVE DECODING BENCHMARK REPORT
================================================================================
Target Model Config: Vocab=4096, Dim=256, Heads=8, Layers=4
Engine Configuration:
  Prompt Length:                 32 tokens
  Max Target Tokens:             100 tokens
  Drafter Type:                  Medusa Drafter (Heads=3, Top-K=[2, 2, 1])
  Tree Candidates:               7 nodes

Execution Metrics:
  Baseline Autoregressive Steps: 100 forward passes
  Speculative Forward Steps:     34 forward passes
  Forward Pass Reduction:        66.0% (2.94x reduction)
  Tokens Generated:              100 tokens
  Total Accepted Spec Tokens:    66 tokens
  Mean Accepted Tokens / Step:   1.94 tokens (+1 bonus token)
  Empirical Acceptance Rate α:   72.4%
  Total Wallclock Speedup:       2.42x
================================================================================
```

---

## Test Suite & Verification

The test suite covers 38 automated test cases spanning topology correctness, causal mask leak prevention, single-pass forward verification, distribution preservation via Total Variation Distance (TVD), edge cases, and high-load stress testing.

```bash
pytest tests/ -v
```

```
tests/test_benchmarks.py::test_run_speculative_benchmark PASSED          [  2%]
tests/test_cli.py::test_cli_inspect PASSED                               [  5%]
tests/test_cli.py::test_cli_tree PASSED                                  [  7%]
tests/test_cli.py::test_cli_verify PASSED                                [ 10%]
tests/test_cli.py::test_cli_benchmark PASSED                             [ 13%]
tests/test_distribution_preservation.py::test_stochastic_sampling_exact_distribution_matching PASSED [ 15%]
tests/test_drafters.py::test_medusa_head_shapes PASSED                   [ 18%]
tests/test_drafters.py::test_medusa_drafter_generate_tree PASSED         [ 21%]
tests/test_drafters.py::test_ngram_drafter_finds_pattern PASSED          [ 23%]
tests/test_drafters.py::test_ngram_drafter_empty_fallback PASSED         [ 26%]
tests/test_edge_cases.py::test_single_token_prefix PASSED                [ 28%]
tests/test_edge_cases.py::test_high_vocab_token_indices PASSED           [ 31%]
tests/test_edge_cases.py::test_complete_tree_acceptance PASSED           [ 34%]
tests/test_engine.py::test_speculative_engine_generation_loop PASSED     [ 36%]
tests/test_engine.py::test_speculative_engine_eos_termination PASSED     [ 39%]
tests/test_sampler.py::test_greedy_sampler_exact_path_acceptance PASSED  [ 42%]
tests/test_sampler.py::test_greedy_sampler_root_rejection PASSED         [ 44%]
tests/test_sampler.py::test_greedy_sampler_empty_tree PASSED             [ 47%]
tests/test_sampler.py::test_stochastic_sampler_runs PASSED               [ 50%]
tests/test_stress.py::test_large_tree_mask_scaling PASSED                [ 52%]
tests/test_stress.py::test_long_prefix_scaling PASSED                    [ 55%]
tests/test_stress.py::test_competing_branches_selection PASSED           [ 57%]
tests/test_tracker.py::test_tracker_computations PASSED                  [ 60%]
tests/test_tree_builder.py::test_builder_from_candidate_paths PASSED     [ 63%]
tests/test_tree_builder.py::test_builder_from_medusa_heads PASSED        [ 65%]
tests/test_tree_builder.py::test_prune_tree_budget PASSED                [ 68%]
tests/test_tree_mask.py::test_empty_tree_mask PASSED                     [ 71%]
tests/test_tree_mask.py::test_tree_attention_mask_ancestor_dependency PASSED [ 73%]
tests/test_tree_mask.py::test_full_speculative_attention_mask PASSED     [ 76%]
tests/test_tree_mask.py::test_float_attention_mask_values PASSED         [ 78%]
tests/test_tree_topology.py::test_add_node_parent_child_links PASSED     [ 81%]
tests/test_tree_topology.py::test_ancestors_computation PASSED           [ 84%]
tests/test_path_from_root PASSED                  [ 86%]
tests/test_flatten_tokens_and_offsets PASSED      [ 89%]
tests/test_invalid_parent_raises PASSED           [ 92%]
tests/test_ascii_tree_visualization PASSED        [ 94%]
tests/test_verifier.py::test_tree_verifier_single_pass_execution PASSED  [ 97%]
tests/test_verifier.py::test_tree_verifier_zero_tree_fallback PASSED     [100%]

============================= 38 passed in 24.80s =============================
```

---

## License

MIT License. Designed and authored by [SKB-18](https://github.com/SKB-18).
