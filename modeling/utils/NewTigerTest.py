import unittest

import torch
from models.tiger import TigerModel
from utils import DEVICE, create_masked_tensor


def create_model():
    return TigerModel(
        sequence_prefix="sequence",
        positive_prefix="positive",
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
            "semantic_sequence.ids": torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 10, 12, 1, 16], device=DEVICE),
            "semantic_sequence.length": torch.tensor([8, 4], device=DEVICE),
            "semantic_positive.ids": torch.tensor([11, 13, 15, 17, 18, 20, 22, 24], device=DEVICE),
            "semantic_positive.length": torch.tensor([4, 4], device=DEVICE),
            "hashed_user.ids": torch.tensor([100, 200], device=DEVICE),
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
            "semantic_sequence.ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=DEVICE),
            "semantic_sequence.length": torch.tensor([8], device=DEVICE),
            "all_semantic_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=DEVICE),
            "semantic_positive.ids": torch.tensor([10, 11, 12, 13], device=DEVICE),
            "semantic_positive.length": torch.tensor([4], device=DEVICE),
            "hashed_user.ids": torch.tensor([100], device=DEVICE),

        }

        outputs = model(inputs)

        assert outputs["decoder_argmax_1"].shape == (1,)
        assert outputs["decoder_scores_4"].shape == (1, 256)

        assert outputs["decoder_argmax_1"].item() in range(256)
        assert outputs["decoder_argmax_4"].item() in range(256)

    def test_prepare_sem_id(self):
        model = create_model()
        total_tokens = 20
        sem_embs_flat = torch.randn(total_tokens, model._embedding_dim)
        lengths = torch.tensor([12, 8])

        # Для декодера берем последние 4 токена каждого примера (sem_id_len=4)
        decoder_embs_list = []
        decoder_lengths = []
        start_idx = 0
        for l in lengths:
            # Берем последние 4 токена
            decoder_embs_list.append(sem_embs_flat[start_idx + l - model._sem_id_len: start_idx + l])
            decoder_lengths.append(model._sem_id_len)
            start_idx += l

        decoder_embs_flat = torch.cat(decoder_embs_list)
        decoder_lengths = torch.tensor(decoder_lengths)

        encoder_emb, encoder_mask, decoder_emb = model._prepare_sem_id_batch(
            encoder_embeddings_flat=sem_embs_flat,
            encoder_lengths=lengths,
            decoder_embeddings_flat=decoder_embs_flat,
            decoder_lengths=decoder_lengths
        )

        sem_embs, _ = create_masked_tensor(sem_embs_flat, lengths)
        decoder_sem_embs, _ = create_masked_tensor(decoder_embs_flat, decoder_lengths)

        # Проверки декодера (первые 5 токенов: [BOS] + последние 4 исходных)
        assert torch.all(decoder_emb[:, 0] == model.bos_embedding)
        for i in range(2):  # Для каждого элемента в батче
            for pos in range(model._sem_id_len):
                # Сравниваем позиции 1-4 в decoder_emb
                expected = decoder_sem_embs[i, pos] + model.sem_id_position_embeddings(torch.tensor(pos))
                assert torch.allclose(decoder_emb[i, pos + 1], expected)

        # Проверки энкодера (первый токен BOS + исходные токены с комбинированными позиционными эмбеддингами)
        assert torch.all(encoder_emb[:, 0] == model.bos_embedding)
        # Для первого элемента (длина=12)
        for pos in range(12):
            # Циклические позиции sem_id (0-3)
            sem_id_pos = pos % model._sem_id_len
            expected = sem_embs[0, pos] + model.position_embeddings(
                torch.tensor(pos)) + model.sem_id_position_embeddings(torch.tensor(sem_id_pos))
            assert torch.allclose(encoder_emb[0, pos + 1], expected)

        # Для второго элемента (длина=8)
        for pos in range(8):
            sem_id_pos = pos % model._sem_id_len
            expected = sem_embs[1, pos] + model.position_embeddings(
                torch.tensor(pos)) + model.sem_id_position_embeddings(torch.tensor(sem_id_pos))
            assert torch.allclose(encoder_emb[1, pos + 1], expected)

        # Проверка паддинга для второго элемента
        for pos in range(8 + 1, 12 + 1):  # Позиции 9-12 в тензоре (индексы 9-12)
            assert torch.allclose(encoder_emb[1, pos], torch.zeros(model._embedding_dim))

    def test_prepare_sem_id(self):
        model = create_model()
        total_tokens = 20
        sem_embs_flat = torch.randn(total_tokens, model._embedding_dim)
        lengths = torch.tensor([12, 8])

        # Для декодера берем последние 4 токена каждого примера
        decoder_embs_list = []
        start_idx = 0
        for l in lengths:
            decoder_embs_list.append(sem_embs_flat[start_idx + l - model._sem_id_len: start_idx + l])
            start_idx += l

        decoder_embs_flat = torch.cat(decoder_embs_list)
        decoder_lengths = torch.full((2,), model._sem_id_len)

        encoder_emb, encoder_mask, decoder_emb = model._prepare_sem_id_batch(
            encoder_embeddings_flat=sem_embs_flat,
            encoder_lengths=lengths,
            decoder_embeddings_flat=decoder_embs_flat,
            decoder_lengths=decoder_lengths
        )

        sem_embs, _ = create_masked_tensor(sem_embs_flat, lengths)
        decoder_sem_embs, _ = create_masked_tensor(decoder_embs_flat, decoder_lengths)

        # Проверка декодера
        assert torch.allclose(decoder_emb[:, 0], model.bos_embedding)
        for i in range(2):
            for pos in range(model._sem_id_len):
                expected = decoder_sem_embs[i, pos] + model.sem_id_position_embeddings(torch.tensor(pos))
                assert torch.allclose(decoder_emb[i, pos + 1], expected)

        # Проверка энкодера
        assert torch.allclose(encoder_emb[:, 0], model.bos_embedding)
        for i, length in enumerate(lengths):
            for pos in range(length):
                sem_id_pos = pos % model._sem_id_len
                expected = sem_embs[i, pos] + model.position_embeddings(
                    torch.tensor(pos)) + model.sem_id_position_embeddings(torch.tensor(sem_id_pos))
                assert torch.allclose(encoder_emb[i, pos + 1], expected)

        # Проверка паддинга для второго элемента
        for pos in range(lengths[1] + 1, lengths[0] + 1):
            assert torch.allclose(encoder_emb[1, pos], torch.zeros(model._embedding_dim))

    def test_only_decoder_data(self):
        model = create_model()
        batch_size = 64
        sem_id_len = 60

        # Энкодер: 64 примера по 4 токена
        encoder_embs_flat = torch.randn(batch_size * sem_id_len, model._embedding_dim)
        encoder_lengths = torch.full((batch_size,), sem_id_len, dtype=torch.long)

        # Декодер: только BOS (пустые данные)
        decoder_embs_flat = torch.empty(0, model._embedding_dim)
        decoder_lengths = torch.zeros(batch_size, dtype=torch.long)

        encoder_emb, encoder_mask, decoder_emb = model._prepare_sem_id_batch(
            encoder_embeddings_flat=encoder_embs_flat,
            encoder_lengths=encoder_lengths,
            decoder_embeddings_flat=decoder_embs_flat,
            decoder_lengths=decoder_lengths
        )

        # Проверяем энкодер
        encoder_sem_embs, _ = create_masked_tensor(encoder_embs_flat, encoder_lengths)
        expected_encoder_emb = torch.zeros(batch_size, sem_id_len + 1, model._embedding_dim, device=DEVICE)
        expected_encoder_emb[:, 0] = model.bos_embedding
        for i in range(batch_size):
            for pos in range(sem_id_len):
                sem_id_pos = pos % model._sem_id_len
                expected_encoder_emb[i, pos + 1] = encoder_sem_embs[i, pos] + model.position_embeddings(
                    torch.tensor(pos)) + model.sem_id_position_embeddings(torch.tensor(sem_id_pos))

        # Проверяем декодер (только BOS)
        expected_decoder_emb = model.bos_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)

        assert torch.allclose(encoder_emb, expected_encoder_emb)
        assert torch.allclose(decoder_emb, expected_decoder_emb)

if __name__ == '__main__':
    unittest.main()

"""
{'user.ids': tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
         14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
         28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
         42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
         56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
         70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
         84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
         98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
        140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
        154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
        168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
        182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
        196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
        210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
        238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
        252, 253, 254, 255]), 'user.length': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'item.ids': tensor([   0,    1,    2,  ..., 2499,  531, 1617]), 'item.length': tensor([ 3,  5,  7,  4,  7, 16,  5,  3, 23,  8,  5,  7,  4,  7,  5,  3,  3,  4,
         7, 35,  6, 11, 10,  7, 18,  7, 49,  7, 15,  7,  3, 49, 10, 11, 12, 14,
         5,  4,  4, 12,  3,  8,  8, 49,  4,  3,  6,  4,  9, 22,  7,  5,  7, 15,
         6,  8,  5,  7,  5,  6,  3,  4, 26,  7,  8,  3,  4,  5,  3, 30,  3,  3,
         7,  8,  3,  3,  4,  3, 16, 14, 23,  9, 13,  9,  9,  7, 49, 12,  4,  8,
         4,  9,  9, 19, 25, 49,  3,  3,  6, 11,  9, 16,  9,  6,  7, 11,  5, 45,
         4,  7, 19,  4,  3, 30,  5,  4,  6, 24, 33,  4,  6,  9,  3,  3,  4, 12,
         3,  3, 10,  6,  5, 19,  3,  4,  3, 49, 16,  4,  6,  3,  5, 11,  7,  4,
         7, 15,  5,  4,  6,  3, 20,  8,  6,  3, 10, 15, 18,  6,  4,  4,  9,  5,
         8,  7,  3, 21,  3, 32, 37,  7,  4, 11,  9, 12,  6,  7,  4, 10, 49,  8,
         7, 12,  7, 10,  7, 15,  4,  5,  3, 11,  6,  7,  3, 10,  3, 17, 14, 46,
         5, 42,  4, 38, 16,  3, 20, 12,  3, 47,  6,  3,  6,  3,  9,  5,  4,  9,
         9,  3,  7, 11,  9,  4,  6,  6, 49,  4, 12,  3,  4, 26,  6,  3,  8,  4,
        13,  3,  3,  3, 49, 13,  3,  7, 22, 21, 16, 15,  7,  5,  3,  4, 11, 13,
         6,  8,  5,  5]), 'labels.ids': tensor([   3,    3,   17,    3,   30,   47,   54,    3,   81,   89,   95,  102,
          82,  114,  120,  124,  128,  133,  141,  177,  183,  194,  155,  140,
         231,  238,  315,  323,  338,  347,  351,  349,  432,  441,  454,  470,
          41,  480,  485,    9,  459,  507,   79,  562,  567,  571,  578,  583,
         592,  612,  459,  622,  629,  642,  649,  656,  659,  667,  673,  680,
         684,  689,  712,  718,  726,  730,  665,  741,  745,  772,  776,  665,
         786,  342,  796,  799,  804,  665,  822,  278,  744,  863,  800,  883,
         892,  899,  954,  961,  965,  974,  978,  852,  991,  933, 1032, 1077,
        1081,  899,  524,  899, 1105,   19, 1125,  421, 1138, 1148, 1151,  952,
          92, 1137, 1205, 1208, 1213, 1237, 1110, 1246, 1250, 1108, 1299, 1304,
         913,  961, 1316, 1320,  890, 1333, 1108, 1337, 1345,  450, 1355, 1372,
        1108, 1379, 1382, 1423, 1438, 1443, 1108, 1108, 1453, 1459, 1465,  307,
        1472,  774, 1484, 1487, 1492, 1496, 1115, 1520, 1525,  240,  324, 1549,
        1560, 1566, 1536, 1573, 1582,   89, 1594, 1599, 1603, 1621, 1262, 1651,
        1682, 1563,  373, 1697, 1703, 1713, 1719, 1725, 1563,  908,  961, 1854,
        1859, 1867, 1874, 1882,  168, 1893, 1899, 1903, 1472, 1913, 1919, 1924,
        1927, 1933, 1474,   23, 1884, 1986, 1367, 1768, 2024, 2047, 2061, 2064,
        2077, 2086, 1787, 2125, 1201, 1557,  562, 2136, 2143, 2147, 2151, 2155,
        2161, 2166, 2171, 2181, 2188, 1520, 1569, 2196,   74, 2278, 2286, 2288,
        2293, 2310, 2317, 1617, 2326, 1617, 2338,   43, 2028, 1325, 1422,  169,
        2406,   19, 2424, 2434, 2446, 1617, 2458, 2462, 2466,  266, 1239,  216,
        2487, 2492, 2495, 2500]), 'labels.length': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'semantic_item.ids': tensor([183,  70, 232,  ..., 250,  32, 155]), 'semantic_item.length': tensor([ 12,  20,  28,  16,  28,  64,  20,  12,  92,  32,  20,  28,  16,  28,
         20,  12,  12,  16,  28, 140,  24,  44,  40,  28,  72,  28, 196,  28,
         60,  28,  12, 196,  40,  44,  48,  56,  20,  16,  16,  48,  12,  32,
         32, 196,  16,  12,  24,  16,  36,  88,  28,  20,  28,  60,  24,  32,
         20,  28,  20,  24,  12,  16, 104,  28,  32,  12,  16,  20,  12, 120,
         12,  12,  28,  32,  12,  12,  16,  12,  64,  56,  92,  36,  52,  36,
         36,  28, 196,  48,  16,  32,  16,  36,  36,  76, 100, 196,  12,  12,
         24,  44,  36,  64,  36,  24,  28,  44,  20, 180,  16,  28,  76,  16,
         12, 120,  20,  16,  24,  96, 132,  16,  24,  36,  12,  12,  16,  48,
         12,  12,  40,  24,  20,  76,  12,  16,  12, 196,  64,  16,  24,  12,
         20,  44,  28,  16,  28,  60,  20,  16,  24,  12,  80,  32,  24,  12,
         40,  60,  72,  24,  16,  16,  36,  20,  32,  28,  12,  84,  12, 128,
        148,  28,  16,  44,  36,  48,  24,  28,  16,  40, 196,  32,  28,  48,
         28,  40,  28,  60,  16,  20,  12,  44,  24,  28,  12,  40,  12,  68,
         56, 184,  20, 168,  16, 152,  64,  12,  80,  48,  12, 188,  24,  12,
         24,  12,  36,  20,  16,  36,  36,  12,  28,  44,  36,  16,  24,  24,
        196,  16,  48,  12,  16, 104,  24,  12,  32,  16,  52,  12,  12,  12,
        196,  52,  12,  28,  88,  84,  64,  60,  28,  20,  12,  16,  44,  52,
         24,  32,  20,  20]), 'semantic_labels.ids': tensor([110, 228,  51,  ...,  59, 211, 244]), 'semantic_labels.length': tensor([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]), 'hashed_user.ids': tensor([ 911,  643,   71, 1300, 1928,  660, 1360,  328,  807,  321,  223, 1394,
        1253,  305,  633,  188, 1280, 1076, 1478,   99,  456,  426, 1772, 1089,
         264,  156,  171, 1210,  766, 1810, 1358,  249,   77, 1351, 1609,   84,
        1295, 1204,  335, 1235,  454,  143,  518, 1803, 1017,  857,  399,  663,
        1621, 1379, 1944,  946, 1313, 1750, 1081,  135,  416,   44, 1930,  632,
        1277, 1965,  599, 1901, 1180, 1491, 1310, 1023, 1793, 1158,  695, 1459,
         123, 1426,  420,  457, 1461, 1139,  168,   52,  689,  896,  286,  988,
        1301,  262, 1035, 1122,  901, 1599,  356,    5,  824,  328,  635, 1876,
        1927,  677, 1369, 1228, 1215,  443, 1969,  642, 1930,  812, 1307, 1788,
        1346, 1171, 1801, 1652,  566,  814, 1808,  585, 1977,  716, 1233,  274,
        1699, 1651,  762,  475, 1892,  190,  985,  567,  226,  409, 1141, 1803,
         572,  210, 1113, 1490,  329,  482,  406, 1288, 1925,   80, 1881,  849,
        1068,  563,  273,  679, 1437,   18, 1503,  434, 1051, 1347,  887,  449,
         535,  340, 1155, 1664, 1817, 1123, 1289,  725, 1110, 1808,   97, 1099,
        1006,  965, 1610,  889,  386,  947,  787, 1893,  445, 1620,  937, 1429,
         862, 1170, 1337, 1273,  800, 1834, 1120,  582, 1317,  232, 1638,  372,
        1594, 1739,  383,   79, 1005,  882, 1579, 1367,  824, 1803, 1340,  600,
        1851, 1966, 1604,  273,   95,  642, 1941,  957,  795, 1824,  953, 1335,
        1269, 1410, 1853, 1277, 1380, 1857, 1791, 1804, 1982, 1749,  108,  905,
         734,  593,  205,  919,  346, 1477, 1657, 1510, 1155, 1511, 1477,  465,
        1093, 1154, 1190, 1731,  123,   52,  199,  233, 1599,  143, 1600,  293,
        1074,  630,  720, 1385]), 'all_semantic_ids': tensor([[183,  70, 232,   6],
        [183, 122,  96, 235],
        [ 52, 170,  54,  27],
        ...,
        [119, 199, 212,  29],
        [119, 200, 102, 124],
        [119, 123,  72, 189]])}
        """