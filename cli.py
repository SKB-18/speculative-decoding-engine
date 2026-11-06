"""
Command-Line Interface for Tree Speculative Decoding Engine.
Subcommands:
  - inspect: Display speculative architecture parameters and mathematical invariants
  - tree: Build and visualize candidate speculative trees (ASCII hierarchy)
  - verify: Execute single-pass parallel verification and display accepted branches
  - benchmark: Run comparative benchmark measuring forward pass reduction and speedup
"""

import argparse
import sys
import json
import torch

from tree.builder import TreeBuilder
from tree.mask import build_tree_attention_mask, build_full_speculative_attention_mask
from drafter.medusa_head import MedusaDrafter
from verifier.tree_verifier import TreeAttentionVerifier
from verifier.sampler import GreedyTreeSampler
from benchmarks.run_benchmark import run_speculative_benchmark, SyntheticTargetModel


def cmd_inspect(args):
    print("================================================================================")
    print("  TREE-STRUCTURED SPECULATIVE DECODING ENGINE (Medusa / EAGLE Style)")
    print("================================================================================")
    print("Core Capabilities:")
    print("  1. Tree Candidate Topology     : Directed rooted trie representing multi-candidate branches")
    print("  2. Causal Tree Attention Mask  : 2D/3D ancestor-only mask preventing cross-branch leakage")
    print("  3. Single-Pass Parallel Verify : Entire tree evaluated in exactly 1 target forward pass")
    print("  4. Acceptance Sampling         : Greedy argmax & Stochastic distribution matching (Leviathan)")
    print("  5. Theoretical Speedup Invariant: S = (1 + mean_accepted_tokens) / target_forward_passes")
    print("--------------------------------------------------------------------------------")
    print(f"Default Configuration:")
    print(f"  - Sampling Mode         : {args.mode}")
    print(f"  - Max Tree Nodes Budget : {args.max_nodes}")
    print(f"  - Branching Strategy    : [2, 2, 2] (3-head Medusa topology)")
    print("================================================================================")


def cmd_tree(args):
    print(f"Building candidate speculative tree with branching factors: {args.branching}...")
    branching = [int(x) for x in args.branching.split(",")]
    
    # Generate mock top-k candidate pools for demonstration
    mock_candidates = []
    for depth, b_factor in enumerate(branching):
        head_pool = [(100 + depth * 10 + i, 0.8 / (i + 1)) for i in range(b_factor)]
        mock_candidates.append(head_pool)

    tree = TreeBuilder.from_medusa_heads(
        head_topk=mock_candidates,
        branching_factors=branching,
        max_nodes=args.max_nodes,
    )

    print(f"\nSpeculative Tree Topology (Total Nodes: {len(tree)}, Max Depth: {tree.get_max_depth()}):")
    print(tree.to_ascii_tree())

    mask = build_tree_attention_mask(tree, as_bool=True)
    print(f"\nTree Attention Mask Shape: {list(mask.shape)}")
    print(f"Mask Active (True) Elements: {mask.sum().item()} / {mask.numel()} "
          f"({(mask.sum().item() / mask.numel()) * 100:.1f}% sparsity)")


def cmd_verify(args):
    print("Executing Single-Pass Parallel Tree Verification...")
    torch.manual_seed(42)
    vocab_size = 50
    hidden_size = 32

    target_model = SyntheticTargetModel(vocab_size=vocab_size, hidden_size=hidden_size)
    prefix_tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    # Build a 2-branch speculative tree
    paths = [[10, 20, 30], [10, 25]]
    tree = TreeBuilder.from_candidate_paths(paths)

    print(f"\nCandidate Tree Structure:")
    print(tree.to_ascii_tree())

    verifier = TreeAttentionVerifier(target_model=target_model, sampling_mode="greedy")
    result = verifier.verify_step(prefix_tokens, tree)

    print("\nVerification Results:")
    print(f"  - Accepted Token IDs  : {result.accepted_token_ids}")
    print(f"  - Accepted Node IDs   : {result.accepted_node_ids}")
    print(f"  - Emitted Bonus Token : {result.bonus_token_id}")
    print(f"  - Total New Tokens    : {result.total_emitted_tokens} ({len(result.total_emitted_tokens)} tokens in 1 forward pass)")


def cmd_benchmark(args):
    print(f"Running comparative benchmarks with {args.tokens} tokens...")
    results = run_speculative_benchmark(num_tokens=args.tokens)
    print("\nBenchmark Summary:")
    print(f"  - Autoregressive Target Passes : {results['autoregressive']['target_forward_passes']}")
    print(f"  - Medusa Speculative Passes    : {results['tree_speculative_medusa']['target_forward_passes']} "
          f"(-{results['tree_speculative_medusa']['forward_pass_reduction_pct']}%)")
    print(f"  - Medusa Acceptance Rate       : {results['tree_speculative_medusa']['acceptance_rate'] * 100:.1f}%")
    print(f"  - Effective Throughput Boost   : {results['tree_speculative_medusa']['effective_throughput_multiplier']}x")


def main():
    parser = argparse.ArgumentParser(description="Tree-Structured Speculative Decoding CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Display architecture parameters")
    p_inspect.add_argument("--mode", default="greedy", choices=["greedy", "stochastic"])
    p_inspect.add_argument("--max-nodes", type=int, default=16)

    # tree
    p_tree = subparsers.add_parser("tree", help="Build and visualize candidate tree")
    p_tree.add_argument("--branching", default="2,2,2", help="Comma-separated branching factors per head")
    p_tree.add_argument("--max-nodes", type=int, default=16)

    # verify
    p_verify = subparsers.add_parser("verify", help="Execute single-pass tree verification")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run comparative speedup benchmark")
    p_bench.add_argument("--tokens", type=int, default=30, help="Tokens to generate")

    args = parser.parse_args()

    if args.subcommand == "inspect":
        cmd_inspect(args)
    elif args.subcommand == "tree":
        cmd_tree(args)
    elif args.subcommand == "verify":
        cmd_verify(args)
    elif args.subcommand == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
