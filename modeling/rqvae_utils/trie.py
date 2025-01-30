import torch
from typing import Optional

from models.rqvae import RqVaeModel

class Item:
    def __init__(
        self, 
        hierarchical_id: tuple[int], 
        raw_item_id: int, 
        residual: torch.Tensor
    ):
        self.hierarchical_id = hierarchical_id
        self.raw_item_id = raw_item_id
        self.residual = residual


class Node:
    def __init__(self,
                 codebook_idx: int,
                 codebook_id: int,
                 parent: Optional['Node'] = None,
                 is_leaf: bool = False):
        self.codebook_idx = codebook_idx
        self.codebook_id = codebook_id
        self.children: dict[int, Node] = {}
        self.leaf_items: list[Item] = []
        self.is_leaf = is_leaf
        self.parent = parent

class HierarchicalTrie:
    def __init__(self, rqvae: RqVaeModel):
        self.root = Node(codebook_idx=-1, codebook_id=-1, parent=None, is_leaf=False)
        self.rqvae = rqvae
    
    def insert(self, item: Item):
        current = self.root
        for level, code_id in enumerate(item.hierarchical_id):
            if code_id not in current.children:
                new_node = Node(
                    codebook_idx=level,
                    codebook_id=code_id,
                    parent=current,
                    is_leaf=False
                )
                current.children[code_id] = new_node
            current = current.children[code_id]

        current.is_leaf = True
        current.leaf_items.append(item)

    def collect_subtree_items(self, node: Node) -> list[Item]:
        result = []
        if node.is_leaf:
            result.extend(node.leaf_items)
        for child in node.children.values():
            result.extend(self.collect_subtree_items(child))
        return result

    def top_k_by_dot_product(
        self, 
        query_vec: torch.Tensor, 
        items: list[Item], 
        k: int
    ) -> list[Item]:
        if not items:
            return []

        dots = []
        for it in items:
            dp = torch.dot(query_vec, it.residual).item()
            dots.append((dp, it))

        dots.sort(key=lambda x: x[0], reverse=True)

        top = [item for (_, item) in dots[:k]]
        return top

    def find_n_closest(self, query_item: Item, n: int) -> list[Item]:
        current = self.root
        matched_path_length = 0

        for level, code_id in enumerate(query_item.hierarchical_id):
            if code_id in current.children:
                current = current.children[code_id]
                matched_path_length += 1
            else:
                break

        node = current

        def try_node(node: Node, query_residual: torch.Tensor, level: int) -> list[Item]:
            if node.is_leaf:
                all_items = node.leaf_items
            else:
                all_items = self.collect_subtree_items(node)

            count_here = len(all_items)

            if count_here == 0:
                if node.parent is None:
                    return []
                return move_up(node, query_residual, level)

            if count_here == n:
                return all_items

            if count_here > n:
                top_n_items = self.top_k_by_dot_product(query_residual, all_items, n)
                return top_n_items

            if node.parent is None:
                return all_items
            else:
                return move_up(node, query_residual, level)

        def move_up(child_node: Node, query_residual: torch.Tensor, level: int) -> list[Item]:
            parent = child_node.parent
            if parent is None:
                return []

            emb = self.rqvae.get_single_embedding(child_node.codebook_idx, child_node.codebook_id)
            new_query_residual = query_residual + emb

            all_parent_items = self.collect_subtree_items(parent)
            count_parent = len(all_parent_items)

            if count_parent == 0:
                if parent.parent is None:
                    return []
                return move_up(parent, new_query_residual, level-1)

            if count_parent >= n:
                adjusted_items = []
                for it in all_parent_items:
                    adj_item_res = it.residual + emb if it in child_node.leaf_items else _adjust_residual_up(it, parent, child_node)
                    new_it = Item(it.hierarchical_id, it.raw_item_id, adj_item_res)
                    adjusted_items.append(new_it)

                top_n_parent = self.top_k_by_dot_product(new_query_residual, adjusted_items, n)
                return top_n_parent

            adjusted_items = []
            for it in all_parent_items:
                adj_item_res = _adjust_residual_up(it, parent, child_node)
                new_it = Item(it.hierarchical_id, it.raw_item_id, adj_item_res)
                adjusted_items.append(new_it)

            # we want to keep them, but also see if we can go further up
            if parent.parent is None:
                return adjusted_items
            else:
                return move_up(parent, new_query_residual, level-1)

        def _adjust_residual_up(it: Item, parent_node: Node, child_node: Node) -> torch.Tensor:
            emb = self.rqvae.get_single_embedding(child_node.codebook_idx, child_node.codebook_id)
            return it.residual + emb

        results = try_node(node, query_item.residual, matched_path_length)
        return results

if __name__ == "__main__":
    trie = HierarchicalTrie()

    emb_dim = 8

    torch.manual_seed(42)
    items = [
        Item((10, 20, 30, 40),  1, torch.randn(emb_dim)),
        Item((10, 20, 30, 40),  2, torch.randn(emb_dim)),  # same path => same leaf
        Item((10, 20, 99, 40),  3, torch.randn(emb_dim)),
        Item((10, 23, 30, 44),  4, torch.randn(emb_dim)),
        Item((10, 23, 30, 45),  5, torch.randn(emb_dim)),
        Item((99,  1,  2,  3),  6, torch.randn(emb_dim))
    ]

    for it in items:
        trie.insert(it)

    query_id = (10, 20, 30, 40)
    query_residual = torch.randn(emb_dim)
    query_item = Item(query_id, raw_item_id=-999, residual=query_residual)

    found_items = trie.find_n_closest(query_item, n=3)

    print("Found items:")
    for fi in found_items:
        print(f"  raw_item_id={fi.raw_item_id}, hierarchical_id={fi.hierarchical_id}")
