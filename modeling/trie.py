class TrieNode:
    def __init__(self):
        self.children = {}
        self.id = None


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tuple_key, id):
        node = self.root
        for number in tuple_key:
            if number not in node.children:
                node.children[number] = TrieNode()
            node = node.children[number]
        node.id = id

    def search(self, tuple_key):
        node = self.root
        for number in tuple_key:
            if number not in node.children:
                return None
            node = node.children[number]
        return node.id

    def _get_all_leaf_ids(self, node):
        result = []
        if node.id is not None:
            result.append(node.id)
        for child in node.children.values():
            result.extend(self._get_all_leaf_ids(child))
        return result

    def search_prefix(self, prefix):
        node = self.root
        for number in prefix:
            if number not in node.children:
                return []
            node = node.children[number]
        return self._get_all_leaf_ids(node)