"""
Tree Builder Primitives.
Constructs SpeculativeTrees from candidate token sequences,
Medusa-style multi-head top-k predictions, or n-gram match dictionaries.
"""

from typing import List, Tuple, Optional, Dict
from .node import SpeculativeTree, SpeculativeTreeNode


class TreeBuilder:
    """
    Factory utilities to build and optimize SpeculativeTrees.
    """

    @staticmethod
    def from_candidate_paths(
        paths: List[List[int]],
        path_probs: Optional[List[float]] = None,
    ) -> SpeculativeTree:
        """
        Builds a prefix tree (Trie) from a list of candidate token sequences.
        Nodes sharing common prefixes are merged into single nodes, maximizing
        verification efficiency and avoiding duplicate computation.

        Example:
            paths = [[10, 20, 30], [10, 20, 40], [10, 50]]
            Result:
                Node 0: 10 (depth 0)
                ├── Node 1: 20 (depth 1)
                │   ├── Node 2: 30 (depth 2)
                │   └── Node 3: 40 (depth 2)
                └── Node 4: 50 (depth 1)
        """
        tree = SpeculativeTree()
        if not paths:
            return tree

        if path_probs is None:
            path_probs = [1.0] * len(paths)

        # Mapping: (depth, token_id, parent_id) -> node_id
        trie_map: Dict[Tuple[int, int, Optional[int]], int] = {}

        for p_idx, path in enumerate(paths):
            prob = path_probs[p_idx]
            parent_id: Optional[int] = None

            for depth, token_id in enumerate(path):
                key = (depth, token_id, parent_id)
                if key in trie_map:
                    node_id = trie_map[key]
                    # Update draft probability if higher
                    if prob > tree.nodes[node_id].draft_prob:
                        tree.nodes[node_id].draft_prob = prob
                    parent_id = node_id
                else:
                    node = tree.add_node(
                        token_id=token_id,
                        depth=depth,
                        parent_id=parent_id,
                        draft_prob=prob,
                    )
                    trie_map[key] = node.node_id
                    parent_id = node.node_id

        return tree

    @staticmethod
    def from_medusa_heads(
        head_topk: List[List[Tuple[int, float]]],
        branching_factors: Optional[List[int]] = None,
        max_nodes: Optional[int] = None,
    ) -> SpeculativeTree:
        """
        Constructs a tree from Medusa heads where head k predicts tokens for offset t + k + 1.
        
        Args:
            head_topk: List of length K, each containing [(token_id, prob), ...] for that head.
            branching_factors: Number of children to expand at each level.
            max_nodes: Upper bound on total tree node count.
        """
        tree = SpeculativeTree()
        K = len(head_topk)
        if K == 0:
            return tree

        if branching_factors is None:
            branching_factors = [len(head_topk[i]) for i in range(K)]

        # Level 0 (depth 0)
        level_0_candidates = head_topk[0][: branching_factors[0]]
        current_layer: List[int] = []

        for token_id, prob in level_0_candidates:
            if max_nodes is not None and len(tree) >= max_nodes:
                break
            node = tree.add_node(token_id=token_id, depth=0, parent_id=None, draft_prob=prob)
            current_layer.append(node.node_id)

        # Subsequent levels
        for depth in range(1, K):
            next_layer: List[int] = []
            candidates = head_topk[depth][: branching_factors[depth]]

            for parent_node_id in current_layer:
                for token_id, prob in candidates:
                    if max_nodes is not None and len(tree) >= max_nodes:
                        break
                    # Child probability is parent_prob * child_prob
                    parent_prob = tree.nodes[parent_node_id].draft_prob
                    combined_prob = parent_prob * prob
                    node = tree.add_node(
                        token_id=token_id,
                        depth=depth,
                        parent_id=parent_node_id,
                        draft_prob=combined_prob,
                    )
                    next_layer.append(node.node_id)

            current_layer = next_layer
            if not current_layer:
                break

        return tree

    @staticmethod
    def prune_tree(tree: SpeculativeTree, max_nodes: int) -> SpeculativeTree:
        """
        Prunes a tree to at most max_nodes, retaining paths with highest probability.
        Guarantees that every retained node has its ancestors retained.
        """
        if len(tree) <= max_nodes:
            return tree

        # Rank nodes by depth (ascending) then draft_prob (descending)
        all_nodes = list(tree.nodes.values())
        all_nodes.sort(key=lambda n: (n.depth, -n.draft_prob))

        retained_ids: Set[int] = set()
        for node in all_nodes:
            # Need to retain this node and all its ancestors
            ancestors = tree.get_ancestors(node.node_id, include_self=True)
            candidate_set = retained_ids.union(ancestors)
            if len(candidate_set) <= max_nodes:
                retained_ids = candidate_set

        # Rebuild new tree with retained nodes
        new_tree = SpeculativeTree()
        id_mapping: Dict[int, int] = {}

        for old_id in sorted(retained_ids):
            old_node = tree.nodes[old_id]
            new_parent = id_mapping.get(old_node.parent_id) if old_node.parent_id is not None else None
            new_node = new_tree.add_node(
                token_id=old_node.token_id,
                depth=old_node.depth,
                parent_id=new_parent,
                draft_prob=old_node.draft_prob,
            )
            id_mapping[old_id] = new_node.node_id

        return new_tree
