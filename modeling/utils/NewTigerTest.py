import unittest

import torch
from models.tiger import NewTiger
from utils import DEVICE


def create_model():
    return NewTiger(
        sequence_prefix="sequence",
        embedding_dim=64,
        num_sids=256,
        num_positions=200,
        num_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=256
    )


class MyTestCase(unittest.TestCase):
    def test_get_last_sem_ids_mask(self):
        model = create_model()
        lengths = torch.tensor([5, 7, 3], device=DEVICE)

        mask = model._get_last_sem_ids_mask(lengths)
        total_tokens = lengths.sum().item()

        assert mask.shape == (total_tokens,)

        expected_positions = [
            [1, 2, 3, 4],  # Для длины 5 индексы 1-4
            [3, 4, 5, 6],  # Для длины 7 индексы 3-6
            [0, 1, 2]  # Для длины 3 все токены
        ]
        flat_expected = torch.tensor([idx for sublist in expected_positions for idx in sublist], device=DEVICE)
        cum_lengths = torch.cat([
            torch.tensor([0], device=lengths.device),
            lengths.cumsum(0)[:-1]
        ])
        offsets = torch.repeat_interleave(cum_lengths, lengths)
        indices = torch.arange(total_tokens, device=lengths.device)
        flat_actual = indices - offsets
        assert torch.sum(mask).item() == len(flat_expected)
        assert torch.all(flat_actual[mask] == flat_expected)

    def test_embed_semantic_tokens(self):
        model = create_model()
        sem_ids = torch.tensor([1, 3, 5, 7, 2, 4, 6, 8], device=DEVICE)

        embeddings = model._embed_semantic_tokens(sem_ids)

        assert embeddings.shape == (8, model._embedding_dim)

        # разные кодбуки используются для разных позиций
        for i in range(8):
            codebook_idx = i % model._sem_id_len
            expected_embed = model.codebook_embeddings[codebook_idx](
                torch.tensor([sem_ids[i]], device=DEVICE)
            )
            assert torch.allclose(embeddings[i], expected_embed)

    def test_position_embeddings(self):
        model = create_model()
        mask = torch.BoolTensor([
            [True, True, False],
            [True, False, False]
        ])

        pos_emb = model._get_position_embeddings(mask)

        assert pos_emb.shape == (2, 3, model._embedding_dim)

        assert torch.all(pos_emb[0, 2] == 0)
        assert torch.all(pos_emb[1, 1:] == 0)

    def test_forward_training(self):
        model = create_model()
        model.train()

        inputs = {
            "sequence.ids": torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 10, 12, 1, 16], device=DEVICE),
            "sequence.length": torch.tensor([8, 4], device=DEVICE),
        }

        outputs = model(inputs)

        assert "decoder_loss_1" in outputs
        assert "decoder_scores_4" in outputs
        assert "decoder_argmax_3" in outputs

        assert outputs["decoder_scores_1"].shape == (2, 256)
        assert outputs["decoder_argmax_4"].shape == (2,)

    def test_autoregressive_decoder(self):
        model = create_model()
        model.eval()

        inputs = {
            "sequence.ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=DEVICE),
            "sequence.length": torch.tensor([8], device=DEVICE),
        }

        outputs = model(inputs)

        assert outputs["decoder_argmax_1"].shape == (1,)
        assert outputs["decoder_scores_4"].shape == (1, 256)

        assert outputs["decoder_argmax_1"].item() in range(256)
        assert outputs["decoder_argmax_4"].item() in range(256)


if __name__ == '__main__':
    unittest.main()
