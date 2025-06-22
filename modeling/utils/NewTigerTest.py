import unittest

import torch
from models.tiger import TigerModel
from utils import DEVICE, create_masked_tensor


def create_model():
    return TigerModel(
        sequence_prefix="sequence",
        embedding_dim=64,
        codebook_size=256,
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
            full_index = (torch.arange(model._sem_id_len, device=DEVICE)[i % model._sem_id_len] * model._codebook_size
                          + sem_ids[i])
            expected_embed = model.codebook_embeddings(full_index)
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

    def test_prepare_sem_id(self):
        model = create_model()
        sem_embs_flat = torch.randn(20, model._embedding_dim)
        lengths = torch.tensor([12, 8])

        encoder_emb, encoder_mask, decoder_emb = model._prepare_sem_id_batch(sem_embs_flat, lengths)
        sem_embs, _ = create_masked_tensor(sem_embs_flat, lengths)

        # Проверка что позиционные эмбеддинги ставятся корректно для декодера

        assert torch.all(decoder_emb[:, 0] == model.bos_embedding)  # роверка что первый токен это bos
        assert torch.all(torch.isclose(decoder_emb[0, 0], model.bos_embedding))
        assert torch.all(torch.isclose(decoder_emb[0, 1],
                                       sem_embs[0, lengths[0] - 4] + model.sem_id_position_embeddings(torch.tensor(0))))
        assert torch.all(torch.isclose(decoder_emb[0, 2],
                                       sem_embs[0, lengths[0] - 3] + model.sem_id_position_embeddings(torch.tensor(1))))
        assert torch.all(torch.isclose(decoder_emb[0, 3],
                                       sem_embs[0, lengths[0] - 2] + model.sem_id_position_embeddings(torch.tensor(2))))
        assert torch.all(torch.isclose(decoder_emb[0, 4],
                                       sem_embs[0, lengths[0] - 1] + model.sem_id_position_embeddings(torch.tensor(3))))
        assert torch.all(torch.isclose(decoder_emb[0, 0], model.bos_embedding))
        assert torch.all(torch.isclose(decoder_emb[1, 1],
                                       sem_embs[1, lengths[1] - 4] + model.sem_id_position_embeddings(torch.tensor(0))))
        assert torch.all(torch.isclose(decoder_emb[1, 2],
                                       sem_embs[1, lengths[1] - 3] + model.sem_id_position_embeddings(torch.tensor(1))))
        assert torch.all(torch.isclose(decoder_emb[1, 3],
                                       sem_embs[1, lengths[1] - 2] + model.sem_id_position_embeddings(torch.tensor(2))))
        assert torch.all(torch.isclose(decoder_emb[1, 4],
                                       sem_embs[1, lengths[1] - 1] + model.sem_id_position_embeddings(torch.tensor(3))))

        # Проверка что позиционные эмбеддинги ставятся корректно для энкодера
        assert torch.all(encoder_emb[:, 0] == model.bos_embedding)
        assert torch.all(torch.isclose(encoder_emb[0, 0], model.bos_embedding))
        assert torch.all(torch.isclose(encoder_emb[0, 1], sem_embs[0, 0] + model.position_embeddings(
            torch.tensor(0)) + model.sem_id_position_embeddings(torch.tensor(0))))
        assert torch.all(torch.isclose(encoder_emb[0, 2], sem_embs[0, 1] + model.position_embeddings(
            torch.tensor(1)) + model.sem_id_position_embeddings(torch.tensor(1))))
        assert torch.all(torch.isclose(encoder_emb[0, 3], sem_embs[0, 2] + model.position_embeddings(
            torch.tensor(2)) + model.sem_id_position_embeddings(torch.tensor(2))))
        assert torch.all(torch.isclose(encoder_emb[0, 4], sem_embs[0, 3] + model.position_embeddings(
            torch.tensor(3)) + model.sem_id_position_embeddings(torch.tensor(3))))
        assert torch.all(torch.isclose(encoder_emb[0, 5], sem_embs[0, 4] + model.position_embeddings(
            torch.tensor(4)) + model.sem_id_position_embeddings(torch.tensor(0))))
        assert torch.all(torch.isclose(encoder_emb[0, 6], sem_embs[0, 5] + model.position_embeddings(
            torch.tensor(5)) + model.sem_id_position_embeddings(torch.tensor(1))))
        assert torch.all(torch.isclose(encoder_emb[0, 7], sem_embs[0, 6] + model.position_embeddings(
            torch.tensor(6)) + model.sem_id_position_embeddings(torch.tensor(2))))
        assert torch.all(torch.isclose(encoder_emb[0, 8], sem_embs[0, 7] + model.position_embeddings(
            torch.tensor(7)) + model.sem_id_position_embeddings(torch.tensor(3))))
        assert torch.all(torch.isclose(encoder_emb[1, 0], model.bos_embedding))
        assert torch.all(torch.isclose(encoder_emb[1, 1], sem_embs[1, 0] + model.position_embeddings(
            torch.tensor(0)) + model.sem_id_position_embeddings(torch.tensor(0))))
        assert torch.all(torch.isclose(encoder_emb[1, 2], sem_embs[1, 1] + model.position_embeddings(
            torch.tensor(1)) + model.sem_id_position_embeddings(torch.tensor(1))))
        assert torch.all(torch.isclose(encoder_emb[1, 3], sem_embs[1, 2] + model.position_embeddings(
            torch.tensor(2)) + model.sem_id_position_embeddings(torch.tensor(2))))
        assert torch.all(torch.isclose(encoder_emb[1, 4], sem_embs[1, 3] + model.position_embeddings(
            torch.tensor(3)) + model.sem_id_position_embeddings(torch.tensor(3))))
        assert torch.all(torch.isclose(encoder_emb[1, 5], torch.zeros(64)))
        assert torch.all(torch.isclose(encoder_emb[1, 6], torch.zeros(64)))
        assert torch.all(torch.isclose(encoder_emb[1, 7], torch.zeros(64)))
        assert torch.all(torch.isclose(encoder_emb[1, 8], torch.zeros(64)))

    def test_only_decoder_data(self):
        sizz = 64
        model = create_model()
        expanded_bos = model.bos_embedding.unsqueeze(0).unsqueeze(0).expand(sizz, -1, -1)

        only_decoder_embs_flat = torch.randn(model._sem_id_len * sizz, model._embedding_dim)
        only_decoder_lengths = torch.tensor([model._sem_id_len for _ in range(sizz)])

        only_decoder_sem_embs, _ = create_masked_tensor(only_decoder_embs_flat, only_decoder_lengths)

        sem_pos_ids = torch.arange(model._sem_id_len, device=DEVICE).expand(sizz, -1)
        sem_pos_emb = model.sem_id_position_embeddings(sem_pos_ids)
        only_decoder_sem_embs += sem_pos_emb

        only_decoder_sem_embs_with_bos = torch.cat([expanded_bos, only_decoder_sem_embs], dim=1)
        only_decoder_encoder_emb, only_decoder_encoder_mask, only_decoder_decoder_emb = model._prepare_sem_id_batch(
            only_decoder_embs_flat, only_decoder_lengths)

        assert torch.all(only_decoder_encoder_emb == expanded_bos).item()
        assert torch.all(only_decoder_decoder_emb == only_decoder_sem_embs_with_bos).item()


if __name__ == '__main__':
    unittest.main()
