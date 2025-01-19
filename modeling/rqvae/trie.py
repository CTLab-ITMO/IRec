from collections import defaultdict
import random

import torch


class Item:
    """
    Represents one data entry with:
      - hierarchical_id: a tuple of int, e.g. (1, 2, 5)
      - residual: a torch.Tensor of shape (emb_dim,)
    """
    def __init__(self, hierarchical_id: tuple[int, ...], residual: torch.Tensor):
        self.hierarchical_id = hierarchical_id
        self.residual = residual

    def __repr__(self):
        return f"Item(id={self.hierarchical_id}, residual={self.residual})"


class TrieNode:
    """
    A node in the Trie.
      - children: dict<int, TrieNode>
      - items: list of Item, non-empty only if this node is
               a terminal for those items' hierarchical_id.
    """
    def __init__(self):
        self.children: dict[int, "TrieNode"] = {}
        self.items: list[Item] = []

    def is_leaf(self) -> bool:
        """
        Return True if this node has no children.
        """
        return len(self.children) == 0


class Trie:
    """
    Trie for storing items keyed by their hierarchical_id.
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, item: Item) -> None:
        """
        Insert an Item into the Trie by walking its hierarchical_id.
        """
        node = self.root
        for idx in item.hierarchical_id:
            if idx not in node.children:
                node.children[idx] = TrieNode()
            node = node.children[idx]
        # This node is now the leaf for item.hierarchical_id
        node.items.append(item)

    def _find_longest_prefix_node(
        self, hierarchical_id: tuple[int, ...]
    ) -> tuple[TrieNode, tuple[int, ...]]:
        """
        Walk the Trie to find the node that matches the longest possible prefix
        of `hierarchical_id`.
        Returns the node and the prefix (as a tuple of int) actually matched.
        """
        node = self.root
        matched_prefix = []

        for idx in hierarchical_id:
            if idx in node.children:
                node = node.children[idx]
                matched_prefix.append(idx)
            else:
                break

        return node, tuple(matched_prefix)

    def _gather_subtree_items(self, node: TrieNode) -> list[Item]:
        """
        Collect all items in the entire subtree rooted at `node`.
        We do a DFS (or BFS) to gather items from every leaf below.
        """
        stack = [node]
        all_items = []

        while stack:
            current = stack.pop()
            # If current is a leaf, it might have items
            if current.items:
                all_items.extend(current.items)
            # Traverse children
            for child in current.children.values():
                stack.append(child)

        return all_items

    def _euclidean_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """
        Euclidean (L2) distance between two torch.Tensors.
        """
        return torch.norm(a - b).item()

    def find_n_closest(self, query_item: Item, n: int) -> list[Item]:
        """
        Find up to `n` closest items to `query_item` by:
          1. Finding the longest existing matching prefix.
          2. First include ALL items from longest matching prefix node
          3. If more slots remain:
             - Go up prefix levels and gather more items
             - Within each prefix level, sort by distance
          4. Never return more than n items total
        """
        # Step 1: Longest prefix node
        node, matched_prefix = self._find_longest_prefix_node(query_item.hierarchical_id)

        # Track items by their prefix length and distance
        collected_by_prefix: dict[int, list[tuple[Item, float]]] = defaultdict(list)
        already_seen = set()

        def gather_and_append(target_node: TrieNode, prefix_length: int):
            """
            Gather items from subtree of `target_node`, compute distances,
            and group them by prefix length.
            """
            subtree_items = self._gather_subtree_items(target_node)
            for it in subtree_items:
                if it not in already_seen:
                    dist = self._euclidean_distance(it.residual, query_item.residual)
                    collected_by_prefix[prefix_length].append((it, dist))
                    already_seen.add(it)

        # First gather items from longest prefix node
        prefix_len = len(matched_prefix)
        gather_and_append(node, prefix_len)

        # If we need more items, move up prefix levels
        while prefix_len > 0 and sum(len(items) for items in collected_by_prefix.values()) < n:
            prefix_len -= 1
            parent = self.root
            for idx in matched_prefix[:prefix_len]:
                parent = parent.children[idx]
            gather_and_append(parent, prefix_len)

        # If still not enough and we're not at root, gather from root
        if prefix_len != 0 and sum(len(items) for items in collected_by_prefix.values()) < n:
            gather_and_append(self.root, 0)

        # Build final result prioritizing longer prefixes
        result = []
        # Start from longest prefix and work down
        for prefix_length in sorted(collected_by_prefix.keys(), reverse=True):
            items = collected_by_prefix[prefix_length]
            # Sort items within this prefix length by distance
            items.sort(key=lambda x: x[1])
            # Add items from this level until we hit n
            remaining_slots = n - len(result)
            result.extend(item for item, _ in items[:remaining_slots])
            if len(result) >= n:
                break

        return result


# ------------------------
# Example Usage:
# ------------------------
if __name__ == "__main__":
    trie = Trie()
    
    # generate random items and stress test
    emb_dim = 4
    for i in range(12000):
        semantic_id = tuple(random.randint(0, 255) for _ in range(4))
        item = Item(semantic_id, torch.rand(emb_dim))
        trie.insert(item)

    # generate query items
    query_items = []
    for i in range(256):
        semantic_id = tuple(random.randint(0, 255) for _ in range(4))
        query_item = Item(semantic_id, torch.rand(emb_dim))
        query_items.append(query_item)
        
    for query_item in query_items:
        closest_items = trie.find_n_closest(query_item, n=20)
        print(f"{len(closest_items)=}")
        # print("Closest items to", query_item, ":\n", closest_items)
