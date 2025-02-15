import json
import time

import torch
from models.rqvae import RqVaeModel
from utils import DEVICE


class Trie:
    def __init__(self, rqvae_model: RqVaeModel):
        self.rqvae_model = rqvae_model
        self.keys = None
        self.prefix_counts = None
        self.residuals_per_level = None
        self.raw_item_ids = None
        self.K = len(self.rqvae_model.codebook_sizes)
        self.total_items = None
        self.embedding_table = torch.stack(
            [cb for cb in self.rqvae_model.codebooks]
        )  # K x codebook_size x embedding_dim

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
        unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

    def compute_keys(self, semantic_ids: torch.Tensor):
        exponents = torch.arange(self.K - 1, -1, -1, device=DEVICE).float()
        base = self.rqvae_model.codebook_sizes[0] ** exponents
        uniq_ids = semantic_ids.float() @ base
        return uniq_ids.int()

    def pad_semantic_ids(self, semantic_ids: torch.Tensor):
        return torch.cat(
            [
                semantic_ids,
                torch.zeros(
                    semantic_ids.shape[0],
                    self.K - semantic_ids.shape[1],
                    dtype=semantic_ids.dtype,
                    device=semantic_ids.device,
                ),
            ],
            dim=1,
        )

    def build_tree_structure(
        self,
        semantic_ids: torch.Tensor,
        residuals: torch.Tensor,
        raw_item_ids: torch.Tensor,
    ):
        """
        Order of semantic ids, residuals, raw_item_ids must be the same (corresponding to same item)
        """
        bs = semantic_ids.shape[0]

        prefix_counts = torch.zeros(bs, self.K + 1, dtype=torch.int64)  # bs x K+1
        prefix_counts[:, 0] = bs

        for i in range(self.K):
            truncated_semantic_ids = semantic_ids[:, : i + 1]
            padded_semantic_ids = self.pad_semantic_ids(truncated_semantic_ids)
            prefix_keys = self.compute_keys(
                padded_semantic_ids
            )  # bs, semantic_ids order
            unique_prefixes, inverse_indices_prefix_counts, prefix_counts_at_level = (
                torch.unique(prefix_keys, return_inverse=True, return_counts=True)
            )  # [1 2 3 3 2] -> [1 2 3] [0 1 2 2 1] [1 2 2]
            current_level_same = prefix_counts_at_level[
                inverse_indices_prefix_counts
            ]  # [1 2 2 2 2]
            prefix_counts[:, i + 1] = current_level_same

        residuals_per_level = self.get_residuals_per_level(
            semantic_ids, residuals
        )  # total_items x K + 1 x embedding_dim

        keys = self.compute_keys(semantic_ids)  # bs, could be collisions

        self.keys = keys
        self.prefix_counts = prefix_counts
        self.residuals_per_level = residuals_per_level
        self.raw_item_ids = raw_item_ids
        self.total_items = len(keys)

    def get_residuals_per_level(
        self,
        semantic_ids: torch.Tensor,
        residuals: torch.Tensor,
    ):
        bs = semantic_ids.shape[0]
        embedding_dim = residuals.shape[1]
        residuals_per_level = torch.zeros(
            bs, self.K + 1, embedding_dim, device=self.embedding_table.device
        )  # bs x K + 1 x embedding_dim

        # TODO think if reverse is needed here
        # i = 3, 2, 1, 0
        for i in range(self.K - 1, -1, -1):
            indices_at_level = semantic_ids[:, i]  # bs
            embeddings_at_level = self.embedding_table[
                i, indices_at_level
            ]  # bs x embedding_dim
            # 1 2 3 4
            residuals_per_level[:, self.K - i, :] = (
                embeddings_at_level + residuals_per_level[:, self.K - i - 1, :]
            )  # [0 first_cumul_emb, second, ..., full_emb]

        # TODO check that residuals_per_level equal at last layer to full embedding of semantic id

        residuals_per_level[:, 0, :] = residuals

        return residuals_per_level  # bs x K + 1 x embedding_dim

    def get_mask_by_prefix(self, prefixes: torch.Tensor, taken_lens: torch.Tensor):
        bs = prefixes.shape[0]
        padded_prefix = self.pad_semantic_ids(prefixes)
        lower_key = self.compute_keys(padded_prefix)  # bs
        upper_key = lower_key + self.rqvae_model.codebook_sizes[0] ** (
            self.K - taken_lens
        )  # bs

        # self.K = 4, prefix_len = 3 => 256 ^ 3 + 256 ^ 2 + 256 ^ 1 + 256 ^ 0
        # need to add 256 ^ 1 to get exclusive upper bound
        # self.K = 4, prefix_len = 2 => 256 ^ 3 + 256 ^ 2 + 256 ^ 1 + 256 ^ 0
        # need to add 256 ^ 2 to get exclusive upper bound
        # self.K = 4, prefix_len = 1 => 256 ^ 3 + 256 ^ 2 + 256 ^ 1 + 256 ^ 0
        # need to add 256 ^ 3 to get exclusive upper bound
        # self.keys.shape = bs, lower_key.shape = bs, upper_key.shape = bs

        assert lower_key.shape[0] == upper_key.shape[0] == bs
        assert self.keys.shape[0] == self.total_items

        mask = (
            (
                self.keys.unsqueeze(0) >= lower_key.unsqueeze(1)
            )  # including prefix [1, 2, 0, 0]
            & (
                self.keys.unsqueeze(0) <= upper_key.unsqueeze(1)
            )  # excluding [1, 3, 0, 0], last [1, 2, 256, 256]
        )

        return mask

    def process_prefixes(self, prefixes: torch.Tensor):
        bs, prefix_len = prefixes.shape
        taken_len = torch.full((bs,), prefix_len, device=DEVICE)
        mask = self.get_mask_by_prefix(prefixes, taken_len)
        # self.keys.unsqueeze(0) = 1 x bs
        # lower_key.unsqueeze(1), upper_key.unsqueeze(1) = bs x 1
        num_items_in_range = (mask).sum(dim=1)
        return num_items_in_range  # bs

    def get_outer_inner_levels(self, semantic_ids: torch.Tensor, items_to_query: int):
        bs, K = semantic_ids.shape
        num_items = torch.stack(
            [self.process_prefixes(semantic_ids[:, : i + 1]) for i in range(K)], dim=1
        )
        num_items: torch.Tensor = torch.cat(
            [
                torch.full(
                    (bs, 1),
                    self.total_items,
                    device=DEVICE,
                ),
                num_items,
            ],
            dim=1,
        )

        # first idx from end where it > items_to_query

        forward_mask = (num_items > items_to_query).int()  # bs x K + 1
        backward_mask = forward_mask.flip(1)  # bs x K + 1
        outer_level = K - torch.argmax(backward_mask, dim=1)  # bs
        inner_level = outer_level + 1  # bs

        # ol & il - how long prefix take => get (> items_to_query & <= items_to_query) items

        assert (outer_level <= K).all()

        return num_items, outer_level, inner_level  # bs x K + 1, bs, bs

    def get_sorted_ids(self, stored_residuals, query_residual, raw_item_ids):
        scores = torch.matmul(stored_residuals, query_residual)  # (guaranteed_len,)
        # Sort guaranteed items by score (descending)
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_ids = raw_item_ids[sorted_indices]

        return sorted_ids

    def get_closest_single(
        self,
        guaranteed_raw_item_ids,  # guaranteed_len
        guaranteed_stored_residuals,  # guaranteed_len x embedding_dim
        guaranteed_query_residual,  # embedding_dim
        left_raw_item_ids,  # left_len
        left_stored_residuals,  # left_len x embedding_dim
        left_query_residual,  # embedding_dim
    ):
        if guaranteed_raw_item_ids.shape[0] == 0:
            guaranteed_sorted_ids = torch.tensor([], device=DEVICE)
        else:
            guaranteed_sorted_ids = self.get_sorted_ids(
                guaranteed_stored_residuals,
                guaranteed_query_residual,
                guaranteed_raw_item_ids,
            )

        left_sorted_ids = self.get_sorted_ids(
            left_stored_residuals, left_query_residual, left_raw_item_ids
        )

        # Concatenate the sorted lists
        result_ids = torch.cat((guaranteed_sorted_ids, left_sorted_ids))

        return result_ids

    def get_closest(
        self,
        outer_masks,
        inner_masks,
        outer_levels,
        inner_levels,
        query_residuals_per_level,
        items_to_query,
    ):
        # K = 3
        # self.residuals_per_level = 0, res, res+emb_2, res+emb_2+emb_1, res+emb_2+emb_1+emb_0 # K + 1
        # query_residuals_per_level = 0, res, res+emb_2, res+emb_2+emb_1, res+emb_2+emb_1+emb_0 # K + 1
        # outer_level = 3 # if K (get by dot only from outer) => inner_mask = outer_mask
        # outer_mask.shape = total_items
        # inner_mask.shape = total_items

        raw_item_ids = []

        for (
            outer_mask,
            inner_mask,
            outer_level,
            inner_level,
            query_residual_per_level,
        ) in zip(
            outer_masks,
            inner_masks,
            outer_levels,
            inner_levels,
            query_residuals_per_level,
        ):
            guaranteed = inner_mask  # guaranteed_len
            left = outer_mask & ~inner_mask  # left_len

            assert guaranteed.sum() + left.sum() == outer_mask.sum()

            # self.residuals_per_level.shape = total_items, K + 1, embedding_dim
            guaranteed_raw_item_ids = self.raw_item_ids[guaranteed]  # guaranteed_len
            guaranteed_stored_residuals = self.residuals_per_level[guaranteed][
                :, -(inner_level + 1)
            ]  # guaranteed_len x embedding_dim
            guaranteed_query_residual = query_residual_per_level[
                -(inner_level + 1)
            ]  # embedding_dim

            left_raw_item_ids = self.raw_item_ids[left]  # left_len
            left_stored_residuals = self.residuals_per_level[left][:, -outer_level]  # left_len x embedding_dim
            left_query_residual = query_residual_per_level[
                -outer_level
            ]  # embedding_dim

            result_ids = self.get_closest_single(
                guaranteed_raw_item_ids,
                guaranteed_stored_residuals,
                guaranteed_query_residual,
                left_raw_item_ids,
                left_stored_residuals,
                left_query_residual,
            )

            raw_item_ids.append(result_ids[:items_to_query])

        return torch.stack(raw_item_ids).int()  # TODO

    def query(
        self, semantic_ids: torch.Tensor, residuals: torch.Tensor, items_to_query: int
    ):
        bs, K = semantic_ids.shape

        assert K == self.K, "Semantic ids must have same number of levels as the trie"

        num_items, outer_levels, inner_levels = self.get_outer_inner_levels(
            semantic_ids, items_to_query
        )  # bs x K + 1, bs, bs

        # print(num_items.shape, outer_levels.shape, inner_levels.shape)
        # print(num_items, outer_levels, inner_levels)

        taken_outer_prefixes = semantic_ids * (
            torch.arange(K, device=DEVICE).expand(bs, K) < outer_levels.unsqueeze(1)
        )
        taken_inner_prefixes = semantic_ids * (
            torch.arange(K, device=DEVICE).expand(bs, K) < inner_levels.unsqueeze(1)
        )

        outer_masks = self.get_mask_by_prefix(
            taken_outer_prefixes, outer_levels
        )  # bs, total_items
        inner_masks = self.get_mask_by_prefix(
            taken_inner_prefixes, inner_levels
        )  # bs, total_items

        # print(inner_masks.shape, outer_masks.shape)
        # print(inner_masks, outer_masks)

        inner_levels_max_mask = inner_levels == self.K + 1
        inner_levels[inner_levels_max_mask] = self.K
        inner_masks[inner_levels_max_mask] = outer_masks[inner_levels_max_mask]

        assert (
            num_items[torch.arange(bs), outer_levels] == outer_masks.sum(dim=1)
        ).all()
        assert (
            num_items[torch.arange(bs), inner_levels] == inner_masks.sum(dim=1)
        ).all()

        assert (outer_masks.sum(dim=1) > items_to_query).all()
        # assert (inner_masks.sum(dim=1) <= items_to_query).all() # can be false if collisions

        assert (inner_masks <= outer_masks).all()

        query_residuals_per_level = self.get_residuals_per_level(
            semantic_ids, residuals
        )

        raw_item_ids = self.get_closest(
            outer_masks,
            inner_masks,
            outer_levels,
            inner_levels,
            query_residuals_per_level,
            items_to_query,
        )

        return raw_item_ids


if __name__ == "__main__":
    embedding_dim = 512  # Embedding size
    config = json.load(open("../configs/train/tiger_train_config.json"))
    config = config["model"]
    rqvae_config = json.load(open(config["rqvae_train_config_path"]))
    rqvae_config["model"]["should_init_codebooks"] = False
    rqvae_model = RqVaeModel.create_from_config(rqvae_config["model"])
    rqvae_model.load_state_dict(
        torch.load(config["rqvae_checkpoint_path"], weights_only=True)
    )
    rqvae_model.eval()

    trie = Trie(rqvae_model)
    alphabet_size = 6

    N = 12101
    K = 3
    # make tensor of size N x K
    # of ([1, 2, 3], [1, 2, 3], [1, 2, 3], ...)
    a = torch.arange(K).repeat(20, 1)
    b = torch.arange(K + 1, K + K + 1).repeat(20, 1)
    semantic_ids = torch.cat([a, b], dim=0)
    residuals = torch.randn(semantic_ids.shape[0], embedding_dim)
    trie.build_tree_structure(
        semantic_ids, residuals, torch.arange(semantic_ids.shape[0])
    )

    items_to_query = 5
    batch_size = 1
    q_semantic_ids = semantic_ids[0].repeat(batch_size, 1)
    # q_semantic_ids = torch.randint(0, alphabet_size, (batch_size, K), dtype=torch.int64)
    q_residuals = torch.randn(batch_size, embedding_dim)

    total_time = 0
    n_exps = 1

    for i in range(n_exps):
        now = time.time()
        item_ids = trie.query(q_semantic_ids, q_residuals, items_to_query)
        print(semantic_ids[item_ids].shape)
        print(q_semantic_ids.shape)
        print(semantic_ids[item_ids] == q_semantic_ids)
        assert item_ids.shape == (batch_size, items_to_query)
        total_time += time.time() - now

    print(f"Time per query: {total_time / n_exps * 1000:.2f} ms")
