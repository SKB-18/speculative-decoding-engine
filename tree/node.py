"""
Speculative Tree Node & Topology Representation.
Encapsulates candidate tokens, parent-child ancestor linkages,
and flattened node arrays for single-pass parallel verification.
"""

from typing import List, Optional, Dict, Set, Tuple
import torch


class SpeculativeTreeNode:
    """
    Represents a single candidate token in a speculative draft tree.
    """

    def __init__(
        self,
        node_id: int,
        token_id: int,
        depth: int,
        parent_id: Optional[int] = None,
        draft_prob: float = 1.0,
    ):
        self.node_id = node_id
        self.token_id = token_id
        self.depth = depth
        self.parent_id = parent_id
        self.draft_prob = draft_prob
        self.children_ids: List[int] = []

    def add_child(self, child_id: int) -> None:
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "token_id": self.token_id,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "draft_prob": self.draft_prob,
            "children_ids": list(self.children_ids),
        }

    def __repr__(self) -> str:
        return (
            f"SpeculativeTreeNode(id={self.node_id}, token={self.token_id}, "
            f"depth={self.depth}, parent={self.parent_id})"
        )


class SpeculativeTree:
    """
    Container for a tree of speculative draft candidates.
    Provides topological analysis, ancestor querying, flattening to PyTorch tensors,
    and path extraction.
    """

    def __init__(self):
        self.nodes: Dict[int, SpeculativeTreeNode] = {}
        self.root_ids: List[int] = []
        self._ancestors_cache: Dict[int, Set[int]] = {}

    def add_node(
        self,
        token_id: int,
        depth: int,
        parent_id: Optional[int] = None,
        draft_prob: float = 1.0,
    ) -> SpeculativeTreeNode:
        """
        Adds a new candidate node to the tree and updates parent-child links.
        """
        node_id = len(self.nodes)
        node = SpeculativeTreeNode(
            node_id=node_id,
            token_id=token_id,
            depth=depth,
            parent_id=parent_id,
            draft_prob=draft_prob,
        )
        self.nodes[node_id] = node

        if parent_id is None:
            self.root_ids.append(node_id)
        else:
            if parent_id not in self.nodes:
                raise ValueError(f"Parent node {parent_id} does not exist in tree.")
            self.nodes[parent_id].add_child(node_id)

        # Invalidate ancestor cache
        self._ancestors_cache.clear()
        return node

    def __len__(self) -> int:
        return len(self.nodes)

    def get_ancestors(self, node_id: int, include_self: bool = True) -> Set[int]:
        """
        Returns all ancestor node IDs for a given node.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in tree.")

        cache_key = (node_id, include_self)
        if cache_key in self._ancestors_cache:
            return self._ancestors_cache[cache_key]

        ancestors = set()
        curr: Optional[int] = node_id if include_self else self.nodes[node_id].parent_id

        while curr is not None:
            ancestors.add(curr)
            curr = self.nodes[curr].parent_id

        self._ancestors_cache[cache_key] = ancestors
        return ancestors

    def get_path_from_root(self, node_id: int) -> List[int]:
        """
        Returns the ordered sequence of node IDs from root down to node_id.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in tree.")

        path = []
        curr: Optional[int] = node_id
        while curr is not None:
            path.append(curr)
            curr = self.nodes[curr].parent_id

        path.reverse()
        return path

    def get_token_sequence(self, node_id: int) -> List[int]:
        """
        Returns the ordered list of token IDs along the path from root to node_id.
        """
        path = self.get_path_from_root(node_id)
        return [self.nodes[nid].token_id for nid in path]

    def get_all_leaves(self) -> List[int]:
        """
        Returns all leaf node IDs (nodes with zero children).
        """
        return [nid for nid, node in self.nodes.items() if len(node.children_ids) == 0]

    def flatten_tokens(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Flattens tree candidate tokens into a 1D tensor [N] ordered by node_id.
        """
        tokens = [self.nodes[i].token_id for i in range(len(self.nodes))]
        return torch.tensor(tokens, dtype=torch.long, device=device)

    def flatten_position_offsets(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Returns the relative depth offset for each node: [N].
        Node with depth 0 has offset 0 (corresponds to offset t+1).
        """
        offsets = [self.nodes[i].depth for i in range(len(self.nodes))]
        return torch.tensor(offsets, dtype=torch.long, device=device)

    def get_max_depth(self) -> int:
        if not self.nodes:
            return 0
        return max(n.depth for n in self.nodes.values()) + 1

    def to_ascii_tree(self) -> str:
        """
        Generates an ASCII visualization of the candidate tree.
        """
        if not self.nodes:
            return "<Empty Tree>"

        lines = []

        def _render(node_id: int, prefix: str = "", is_last: bool = True):
            node = self.nodes[node_id]
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}[Node {node.node_id}] Token: {node.token_id} (d={node.depth}, p={node.draft_prob:.2f})")
            
            new_prefix = prefix + ("    " if is_last else "│   ")
            child_count = len(node.children_ids)
            for idx, cid in enumerate(node.children_ids):
                _render(cid, new_prefix, idx == child_count - 1)

        for i, root_id in enumerate(self.root_ids):
            _render(root_id, is_last=(i == len(self.root_ids) - 1))

        return "\n".join(lines)
