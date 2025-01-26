from functools import lru_cache

import torch


class Item:
    """
    Represents an item with:
      - hierarchical_id: a tuple of ints
      - residual: a torch.Tensor of shape (emb_dim,)
    """
    def __init__(self, hierarchical_id: tuple[int, ...], residual: torch.Tensor):
        self.hierarchical_id = hierarchical_id
        self.residual = residual

    def __repr__(self):
        return f"Item(hier_id={self.hierarchical_id}, resid_shape={tuple(self.residual.shape)})"


class TrieNode:
    """
    A node in the trie.
      - children: dict(int -> TrieNode)
      - items: list of Items (only non-empty if this node is the end of some hierarchical_id(s))
    """
    def __init__(self):
        self.children: dict[int, TrieNode] = {}
        self.items: list[Item] = []  # collisions end up here if they share the same path


class Trie:
    """
    The trie structure that can:
      1. Insert an Item by its hierarchical_id.
      2. Find up to n closest items to a query Item by the described prefix-truncation + dot-product logic.
    """
    def __init__(self):
        self.root = TrieNode()
        # Create instance-specific cached methods
        self._cached_gather_subtree_items = self._gather_subtree_items
        self._cached_dot_scores = self._dot_scores
        # self._cached_gather_subtree_items = lru_cache(maxsize=1024)(self._gather_subtree_items)
        # self._cached_dot_scores = lru_cache(maxsize=1024)(self._dot_scores)

    def insert(self, item: Item) -> None:
        """
        Insert an Item into the trie following item.hierarchical_id.
        """
        current_node = self.root
        for idx in item.hierarchical_id:
            if idx not in current_node.children:
                current_node.children[idx] = TrieNode()
            current_node = current_node.children[idx]
        # At the leaf, store the item
        current_node.items.append(item)

    def _gather_subtree_items(self, node: TrieNode) -> tuple[Item, ...]:
        """
        Collect all items in the sub-tree rooted at 'node'.
        """
        collected = []
        stack = [node]
        while stack:
            cur = stack.pop()
            collected.extend(cur.items)
            for child_node in cur.children.values():
                stack.append(child_node)
        return tuple(collected)

    def _dot_scores(
        self, 
        query_residual: torch.Tensor, 
        candidates: tuple[Item, ...]
    ) -> tuple[tuple[Item, float], ...]:
        """
        Compute dot-product scores between query_residual and each candidate's residual.
        Return tuple of (candidate_item, score) pairs.
        """
        scored = []
        for c in candidates:
            score = torch.dot(query_residual, c.residual).item()
            scored.append((c, score))
        return tuple(scored)

    def find_closest(self, query_item: Item, n: int) -> list[Item]:
        """
        Find up to n closest items to 'query_item' by the described rules:
          1) Find longest existing matching prefix.
          2) Gather sub-tree items:
             - If >= n items, pick top n by dot product.
             - Else, move to parent, gather sub-tree, etc.
          3) If root is reached, return all if < n, else top n by dot product.
        """
        # 1) Descend as far as possible
        path_stack = [(self.root, None)]  # (node, parent_node) pairs for easy upward climb
        current_node = self.root

        # Traverse hierarchy while possible
        for idx in query_item.hierarchical_id:
            if idx in current_node.children:
                parent_node = current_node
                current_node = current_node.children[idx]
                path_stack.append((current_node, parent_node))
            else:
                # Can't go deeper, break
                break

        # Now path_stack[-1][0] is the deepest node we matched.
        # We'll climb up if needed.
        while path_stack:
            node, parent = path_stack[-1]
            subtree_items = self._cached_gather_subtree_items(node)
            if len(subtree_items) >= n:
                # We can pick the top n by dot product
                scored = self._cached_dot_scores(query_item.residual, subtree_items)
                # Sort descending by score
                scored = sorted(scored, key=lambda x: x[1], reverse=True)
                return [itm for itm, _ in scored[:n]]
            else:
                # Not enough items: move up one level
                # (pop the current node off the stack and keep going)
                path_stack.pop()
                if not path_stack:
                    # We are at the root (the last pop).
                    # If still not enough items, we simply return everything we have from the root
                    # or if root has more than n, we pick top n.
                    scored = self._cached_dot_scores(query_item.residual, subtree_items)
                    scored = sorted(scored, key=lambda x: x[1], reverse=True)
                    return [itm for itm, _ in scored[:n]]
                # Otherwise, continue in the loop (which means gather from the parent's node)

        # Safety net (should never get here logically, but just in case):
        return []


# ---------------------------
# Usage example (toy):
if __name__ == "__main__":
    # Suppose emb_dim = 3 for example
    trie = Trie()

    # Insert a few items
    items_to_insert = [
        Item((1, 2, 3), torch.tensor([1.0, 0.5, 0.2])),
        Item((1, 2, 3), torch.tensor([1.1, 0.4, 0.0])),  # collision on same path
        Item((1, 2, 4), torch.tensor([0.9, 0.9, 0.9])),
        Item((1, 5),     torch.tensor([-0.1, 0.2, 1.0])),
        Item((2,),       torch.tensor([0.3, 0.3, 0.3])),
    ]
    for it in items_to_insert:
        trie.insert(it)

    # Query item
    query = Item((1, 2, 3, 99), torch.tensor([1.0, 1.0, 1.0]))  # (1,2,3,99) partially matches deeper
    closest = trie.find_closest(query, n=3)
    print("Closest items:")
    for c in closest:
        print(c)
