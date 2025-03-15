import json

import numpy as np
import torch

from models import RqVaeModel
from utils import DEVICE


def test(a, b):
    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=0)
    norm_a = torch.norm(a, p=2)
    norm_b = torch.norm(b, p=2)
    l2_dist = torch.norm(a - b, p=2) / (norm_a + norm_b + 1e-8)
    return cos_sim, l2_dist


if __name__ == "__main__":
    config = json.load(open("../configs/train/tiger_train_config.json"))
    config = config["model"]
    rqvae_config = json.load(open(config["rqvae_train_config_path"]))
    rqvae_config["model"]["should_init_codebooks"] = False
    rqvae_model = RqVaeModel.create_from_config(rqvae_config["model"]).to(DEVICE)
    rqvae_model.load_state_dict(
        torch.load(config["rqvae_checkpoint_path"], weights_only=True)
    )
    df = torch.load(config["embs_extractor_path"], weights_only=False)
    embeddings_array = np.stack(df["embeddings"].values)
    tensor_embeddings = torch.tensor(
        embeddings_array, dtype=torch.float32, device=DEVICE
    )
    inputs = {"embeddings": tensor_embeddings}

    rqvae_model.eval()
    sem_ids, residuals = rqvae_model.forward(inputs)
    scores = residuals.detach()
    print(torch.norm(residuals, p=2, dim=1).median())
    for i, codebook in enumerate(rqvae_model.codebooks):
        scores += codebook[sem_ids[:, i]].detach()
    decoder_output = rqvae_model.decoder(scores.detach()).detach()

    a = tensor_embeddings[0]
    b = decoder_output[0]
    cos_sim, l2_dist = test(a, b)
    print("косинусное расстояние", cos_sim)
    print("евклидово расстояние", l2_dist)

    cos_sim = torch.nn.functional.cosine_similarity(
        tensor_embeddings, decoder_output, dim=1
    )
    print("косинусное расстояние", cos_sim.mean(), cos_sim.min(), cos_sim.max())

    norm_a = torch.norm(tensor_embeddings, p=2, dim=1)
    norm_b = torch.norm(decoder_output, p=2, dim=1)
    l2_dist = torch.norm(decoder_output - tensor_embeddings, p=2, dim=1) / (
        norm_a + norm_b
    )
    print("евклидово расстояние", l2_dist.median(), l2_dist.min(), l2_dist.max())\

    temp = (torch.norm(tensor_embeddings - decoder_output, dim=1) / norm_a)
    print("||x_hat - x|| / ||x||", temp.median(), temp.min(), temp.max())
    print("||x_hat|| / ||x||", (norm_b / norm_a).median(), (norm_b / norm_a).min(), (norm_b / norm_a).max())
"""
Spherical
tensor(0.0497, grad_fn=<MedianBackward0>)
косинусное расстояние tensor(0.9612)
евклидово расстояние tensor(0.1431)
косинусное расстояние tensor(0.9478) tensor(0.8512) tensor(0.9935)
евклидово расстояние tensor(0.1625) tensor(0.0570) tensor(0.2835)
||x_hat - x|| / ||x|| tensor(0.3158) tensor(0.1142) tensor(0.5248)
||x_hat|| / ||x|| tensor([0.9352, 0.9467, 0.9293,  ..., 0.9218, 0.9605, 0.9425])

обычное
tensor(0.0462, grad_fn=<MedianBackward0>)
косинусное расстояние tensor(0.9615)
евклидово расстояние tensor(0.1429)
косинусное расстояние tensor(0.9459) tensor(0.8376) tensor(0.9932)
евклидово расстояние tensor(0.1653) tensor(0.0582) tensor(0.2994)
||x_hat - x|| / ||x|| tensor(0.3209) tensor(0.1167) tensor(0.5465)
||x_hat|| / ||x|| tensor([0.9329, 0.9420, 0.9346,  ..., 0.9207, 0.9589, 0.9419])
"""

"""
Параметры пети
emb 768, 800 итераций
(12101, 768)
tensor(0.0342, grad_fn=<MedianBackward0>)
косинусное расстояние tensor(0.9492)
евклидово расстояние tensor(0.1642)
косинусное расстояние tensor(0.9078) tensor(0.7222) tensor(0.9839)
евклидово расстояние tensor(0.2187) tensor(0.0920) tensor(0.4088)
||x_hat - x|| / ||x|| tensor(0.4138) tensor(0.1802) tensor(0.6918)
||x_hat|| / ||x|| tensor(0.8940) tensor(0.6922) tensor(0.9783)

tensor(0.0302, grad_fn=<MedianBackward0>)
косинусное расстояние tensor(0.9424)
евклидово расстояние tensor(0.1734)
косинусное расстояние tensor(0.8977) tensor(0.7033) tensor(0.9811)
евклидово расстояние tensor(0.2303) tensor(0.0990) tensor(0.4179)
||x_hat - x|| / ||x|| tensor(0.4349) tensor(0.1944) tensor(0.7110)
||x_hat|| / ||x|| tensor(0.8902) tensor(0.6925) tensor(0.9884)

"""