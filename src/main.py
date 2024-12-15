import torch
import typing
import random
import os

from rqvae import RQVAE

device = torch.device("cuda")


def get_cb_tuples(embeddings):
    ind_lists = []
    for cb in rqvae.codebooks:
        dist = torch.cdist(rqvae.encoder(embeddings), cb)
        ind_lists.append(dist.argmin(dim=-1).cpu().numpy())

    return zip(*ind_lists)


def search_similar_items(items_with_tuples):
    random.shuffle(items_with_tuples)
    clust2search = (585,)
    cnt = 0
    for item, clust_tuple in items_with_tuples:
        if clust_tuple[: len(clust2search)] == clust2search:
            print(item, clust_tuple)
            cnt += 1
        if cnt >= 5:
            break


# TODO: add T5 sentence construction from huggingface


embs = {"embedding": []}

rqvae = RQVAE(
    input_dim=200,
    hidden_dim=128,
    beta=0.25,
    codebook_sizes=[256] * 4,
    should_init_codebooks=False,
    should_reinit_unused_clusters=False,
).to(device)

rqvae.forward(embs)
