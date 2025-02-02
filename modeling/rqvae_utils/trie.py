import torch

def unique_with_index(x, dim=None):
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

def compute_unique_ids(semantic_ids: torch.Tensor):
    K = semantic_ids.shape[1]
    exponents = torch.arange(K - 1, -1, -1, dtype=torch.int64)
    base = 256 ** exponents # TODO don't hardcode 256
    uniq_ids = torch.einsum('nc,c->n', semantic_ids, base)
    return uniq_ids

def build_tree_structure(semantic_ids: torch.Tensor, residuals: torch.Tensor, embedding_table: torch.Tensor):
    bs, K = semantic_ids.shape
    embedding_dim = residuals.shape[1]

    prefix_counts = torch.zeros(len(semantic_ids), K + 1, dtype=torch.int64) # num_unique x K+1
    prefix_counts[:, 0] = bs

    for i in range(K):
        prefix_keys = compute_unique_ids(semantic_ids[:, :i+1]) # bs, semantic_ids order
        unique_prefixes, inverse_indices_prefix_counts, prefix_counts_at_level = torch.unique(prefix_keys, return_inverse=True, return_counts=True) # num_unique_prefix, num_unique_prefix
        current_level_same = prefix_counts_at_level[inverse_indices_prefix_counts]
        prefix_counts[:, i + 1] = current_level_same
    
    # TODO print prefix_counts

    residuals_per_level = torch.zeros(len(semantic_ids), K + 1, embedding_dim) # num_unique x K+1 x embedding_dim

    for i in range(K - 1, -1, -1):
        indices_at_level = semantic_ids[:, i] # bs
        embeddings_at_level = embedding_table[i, indices_at_level] # bs x embedding_dim
        residuals_per_level[:, K - i, :] = embeddings_at_level + residuals_per_level[:, K - i - 1, :]

    large_uniq_ids = compute_unique_ids(semantic_ids) # bs, could be collisions
    _, unique_indicies = unique_with_index(large_uniq_ids) # TODO check with semantic_ids (must be same)
    
    prefix_counts = prefix_counts[unique_indicies]
    residuals_per_level = residuals_per_level[unique_indicies]
    unique_ids = large_uniq_ids[unique_indicies]

    sorted_indices = torch.argsort(unique_ids)
    sorted_uniq_ids = unique_ids[sorted_indices]
    sorted_prefix_counts = prefix_counts[sorted_indices]
    sorted_residuals = residuals_per_level[sorted_indices]

    return sorted_uniq_ids, sorted_prefix_counts, sorted_residuals


bs = 12101  # Batch size
K = 4    # Length of semantic_id
embedding_dim = 512  # Embedding size

# Generate random semantic IDs in range [0, 255]
semantic_ids = torch.randint(0, 256, (bs, K), dtype=torch.int64)

print(f"{semantic_ids=}")

# Random residuals
residuals = torch.randn(bs, embedding_dim)

# Example embedding table (num_levels=K, each level has 256 embeddings of size embedding_dim)
embedding_table = torch.randn(K, 256, embedding_dim)

# Build tree structure
sorted_uniq_ids, sorted_prefix_counts, sorted_residuals = build_tree_structure(semantic_ids, residuals, embedding_table)

# Print results
print("Sorted Unique IDs:", sorted_uniq_ids)
print("Sorted Prefix Counts:", sorted_prefix_counts)
print("Sorted Residuals:", sorted_residuals)