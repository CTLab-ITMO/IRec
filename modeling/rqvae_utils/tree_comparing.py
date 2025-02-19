import json
import os
import time

import psutil
import torch

from models.rqvae import RqVaeModel
from rqvae_utils import Trie, SimplifiedTree, Tree
from utils import DEVICE


def memory_stats(k):
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 ** 2
    print(f"{k}. Использование памяти: {memory_usage:.2f} MB")


def calc_sid(sid, codebook_size):
    res = sid[-1]
    for i in range(1, sid.shape[0]):
        res += sid[-i - 1] * (codebook_size ** i)
    return res


def stats(query_sem_id, codebook_size, sids, item_ids):
    for sem_id, ids in zip(query_sem_id.tolist(), item_ids.tolist()):
        print(calc_sid(torch.tensor(sem_id), codebook_size))
        print(sids[torch.tensor(ids)][:10])


if __name__ == "__main__":
    embedding_dim = 64  # Embedding size
    config = json.load(open("../configs/train/tiger_train_config.json"))
    config = config["model"]
    rqvae_config = json.load(open(config["rqvae_train_config_path"]))
    rqvae_config["model"]["should_init_codebooks"] = False
    rqvae_model = RqVaeModel.create_from_config(rqvae_config["model"]).to(DEVICE)
    rqvae_model.load_state_dict(
        torch.load(config["rqvae_checkpoint_path"], weights_only=True)
    )
    rqvae_model.eval()

    emb_table = torch.stack(
        [cb for cb in rqvae_model.codebooks]
    ).to(DEVICE)

    trie = Trie(rqvae_model)
    tree = Tree(emb_table, DEVICE)
    simplified_tree = SimplifiedTree(emb_table, DEVICE)
    simplified_tree_wr = SimplifiedTree(emb_table, DEVICE)
    alphabet_size = 10

    N = 12101
    K = 3

    semantic_ids = torch.randint(0, alphabet_size, (N, K), dtype=torch.int64).to(DEVICE)
    residuals = torch.randn(N, embedding_dim).to(DEVICE)
    item_ids = torch.arange(5, N + 5).to(DEVICE)
    print(residuals[0])

    now = time.time()
    trie.build_tree_structure(semantic_ids, residuals, item_ids)
    print(f"Time for trie init: {(time.time() - now) * 1000:.2f} ms")

    now = time.time()
    tree.build_tree_structure(semantic_ids, residuals, item_ids)
    print(f"Time for tree init: {(time.time() - now) * 1000:.2f} ms")

    now = time.time()
    simplified_tree.build_tree_structure(semantic_ids, residuals, item_ids)
    print(f"Time for simplified tree init: {(time.time() - now) * 1000:.2f} ms")

    now = time.time()
    simplified_tree_wr.build_tree_structure(semantic_ids, residuals, item_ids, False)
    print(f"Time for simplified tree  without residuals init: {(time.time() - now) * 1000:.2f} ms")

    full_embeddings = tree.calculate_full(semantic_ids, residuals).sum(1)
    print(torch.all((full_embeddings == simplified_tree.full_embeddings) == True))

    items_to_query = 20
    batch_size = 256
    q_semantic_ids = torch.randint(0, alphabet_size, (batch_size, K), dtype=torch.int64, device=DEVICE)
    q_residuals = torch.randn(batch_size, embedding_dim).to(DEVICE)

    total_time = 0
    n_exps = 1

    memory_stats(1)
    for i in range(n_exps):
        now = time.time()
        item_ids = trie.query(q_semantic_ids, q_residuals, items_to_query)
        total_time += time.time() - now
        stats(q_semantic_ids[:1], 256, tree.sids, item_ids[:1])

    print(f"Time per query: {total_time / n_exps * 1000:.2f} ms")

    memory_stats(2)

    for i in range(n_exps):
        now = time.time()
        simplified_tree_ids = simplified_tree.query(q_semantic_ids, items_to_query)
        total_time += time.time() - now
        stats(q_semantic_ids[:1], 256, tree.sids, simplified_tree_ids[:1])

    print(f"Time per query: {total_time / n_exps * 1000:.2f} ms")

    memory_stats(3)

    for i in range(n_exps):
        now = time.time()
        simplified_tree_ids = simplified_tree_wr.query(q_semantic_ids, items_to_query)
        total_time += time.time() - now
        stats(q_semantic_ids[:1], 256, tree.sids, simplified_tree_ids[:1])

    print(f"Time per query: {total_time / n_exps * 1000:.2f} ms")

    memory_stats(4)

    for i in range(n_exps):
        now = time.time()
        tree_ids = tree.query(q_semantic_ids, q_residuals, items_to_query)
        total_time += time.time() - now
        stats(q_semantic_ids[:1], 256, tree.sids, tree_ids[:1])

    print(f"Time per query: {total_time / n_exps * 1000:.2f} ms")

    memory_stats(5)
