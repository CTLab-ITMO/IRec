import json
import torch

from models.rqvae import RqVaeModel


class Trie:
    def __init__(self, rqvae_model: RqVaeModel):
        self.rqvae_model = rqvae_model
        self.keys = None
        self.prefix_counts = None
        self.residuals = None
        self.raw_item_ids = None
        
    def unique_with_index(self, x, dim=None):
        """Unique elements of x and indices of those unique elements
        https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

        e.g.

        unique(tensor([
            [1, 2, 3],
            [1, 2, 4],
            [1, 2, 3],
            [1, 2, 5]
        ]), dim=0)
        => (tensor([[1, 2, 3],
                    [1, 2, 4],
                    [1, 2, 5]]),
            tensor([0, 1, 3]))
        """
        unique, inverse = torch.unique(
            x, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                            device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

    def compute_keys(self, semantic_ids: torch.Tensor):
        K = semantic_ids.shape[1]
        exponents = torch.arange(K - 1, -1, -1, dtype=torch.int64)
        base = self.rqvae_model.codebook_sizes[0] ** exponents # TODO don't hardcode 256
        uniq_ids = torch.einsum('nc,c->n', semantic_ids, base)
        return uniq_ids

    def build_tree_structure(self, semantic_ids: torch.Tensor, residuals: torch.Tensor, raw_item_ids: torch.Tensor):
        embedding_table = torch.stack([cb for cb in self.rqvae_model.codebooks])
        
        print(f"{embedding_table.shape=}")
        
        bs, K = semantic_ids.shape
        embedding_dim = residuals.shape[1]

        prefix_counts = torch.zeros(bs, K + 1, dtype=torch.int64) # bs x K+1
        prefix_counts[:, 0] = bs

        for i in range(K):
            truncated_semantic_ids = semantic_ids[:, :i+1]
            padded_semantic_ids = torch.cat([truncated_semantic_ids, torch.zeros(bs, K - i - 1, dtype=torch.int64)], dim=1)
            prefix_keys = self.compute_keys(padded_semantic_ids) # bs, semantic_ids order
            unique_prefixes, inverse_indices_prefix_counts, prefix_counts_at_level = torch.unique(prefix_keys, return_inverse=True, return_counts=True)
            current_level_same = prefix_counts_at_level[inverse_indices_prefix_counts]
            prefix_counts[:, i + 1] = current_level_same
        
        # TODO print prefix_counts

        residuals_per_level = torch.zeros(bs, K + 1, embedding_dim) # bs x K+1 x embedding_dim

        for i in range(K - 1, -1, -1):
            indices_at_level = semantic_ids[:, i] # bs
            embeddings_at_level = embedding_table[i, indices_at_level] # bs x embedding_dim
            residuals_per_level[:, K - i, :] = embeddings_at_level + residuals_per_level[:, K - i - 1, :]

        keys = self.compute_keys(semantic_ids) # bs, could be collisions
        # _, unique_indicies = self.unique_with_index(large_uniq_ids) # TODO check with semantic_ids (must be same)
        
        # prefix_counts = prefix_counts[unique_indicies]
        # residuals_per_level = residuals_per_level[unique_indicies]
        # unique_ids = large_uniq_ids[unique_indicies]
        
        self.keys = keys
        self.prefix_counts = prefix_counts
        self.residuals = residuals_per_level
        self.raw_item_ids = raw_item_ids

        # sorted_indices = torch.argsort(keys)

        # self.sorted_keys = keys[sorted_indices]
        # self.sorted_prefix_counts = prefix_counts[sorted_indices]
        # self.sorted_residuals = residuals_per_level[sorted_indices]
        # self.sorted_raw_item_ids = raw_item_ids[sorted_indices]
        
    def process_prefixes(self, prefixes: torch.Tensor):
        bs, prefix_len = prefixes.shape
        padded_prefix = torch.cat([prefixes, torch.zeros(bs, len(self.rqvae_model.codebook_sizes) - prefix_len, dtype=prefixes.dtype)], dim=1)
        lower_key = self.compute_keys(padded_prefix) # bs
        upper_key = lower_key + self.rqvae_model.codebook_sizes[0] ** (len(self.rqvae_model.codebook_sizes) - prefix_len) # bs
        num_items_in_range = ((self.keys.unsqueeze(0) >= lower_key.unsqueeze(1)) & (self.keys.unsqueeze(0) < upper_key.unsqueeze(1))).sum(dim=1)
        return num_items_in_range # bs
        
        
    def query(self, semantic_ids: torch.Tensor, residuals: torch.Tensor, item_to_query: int):
        bs, K = semantic_ids.shape
        num_items = torch.stack([self.process_prefixes(semantic_ids[:, :i+1]) for i in range(K)], dim=1)
        # max idx for each row where num_items > item_to_query
        forward_mask = (num_items > item_to_query).int()
        backward_mask = forward_mask.flip(1)
        outer_level = K - 1 - torch.argmax(backward_mask, dim=1)
        inner_level = outer_level + 1
        
        taken_outer_prefixes = semantic_ids * (torch.arange(K).expand(bs, K) < outer_level.unsqueeze(1))
        taken_inner_prefixes = semantic_ids * (torch.arange(K).expand(bs, K) < inner_level.unsqueeze(1))
        
        outer_lower_prefix_keys = self.compute_keys(taken_outer_prefixes)
        outer_upper_prefix_keys = outer_lower_prefix_keys + self.rqvae_model.codebook_sizes[0] ** (K - 1 - outer_level)
        
        print(outer_lower_prefix_keys)
        
        outer_mask = (self.keys.unsqueeze(0) >= outer_lower_prefix_keys.unsqueeze(1)) & (self.keys.unsqueeze(0) < outer_upper_prefix_keys.unsqueeze(1))
        
        inner_lower_prefix_keys = self.compute_keys(taken_inner_prefixes)
        inner_upper_prefix_keys = inner_lower_prefix_keys + self.rqvae_model.codebook_sizes[0] ** (K - 1 - inner_level)
        
        inner_mask = (self.keys.unsqueeze(0) >= inner_lower_prefix_keys.unsqueeze(1)) & (self.keys.unsqueeze(0) < inner_upper_prefix_keys.unsqueeze(1))
        
        assert (inner_mask <= outer_mask).all() # TODO fix this
        
        return outer_mask, inner_mask


if __name__ == "__main__":
    embedding_dim = 512  # Embedding size
    config = json.load(open("../configs/train/tiger_train_config.json"))
    config = config["model"]
    rqvae_config = json.load(open(config["rqvae_train_config_path"]))
    rqvae_config["model"]["should_init_codebooks"] = False
    rqvae_model = RqVaeModel.create_from_config(rqvae_config['model'])
    rqvae_model.load_state_dict(torch.load(config['rqvae_checkpoint_path'], weights_only=True))
    rqvae_model.eval()
    
    trie = Trie(rqvae_model)
    

    N = 100
    K = 3
    semantic_ids = torch.randint(0, 4, (N, K), dtype=torch.int64)
    residuals = torch.randn(N, embedding_dim)
    trie.build_tree_structure(semantic_ids, residuals, torch.arange(N))
    
    query_num = 10
    q_semantic_ids = torch.randint(0, 4, (query_num, K), dtype=torch.int64)
    q_residuals = torch.randn(query_num, embedding_dim)

    a, b  = trie.query(q_semantic_ids, q_residuals, 10)
    # print(f"{a=}")
    # print(f"{b=}")
    print(f"{a.shape=}")
    print(f"{b.shape=}")
