import torch
import numpy as np
import pytest
from transforms import AddWeightedCooccurrenceEmbeddingsVectorized

def test_add_weighted_cooccurrence_embeddings():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cpu') # Тестируем на CPU для простоты

    print("\n" + "="*80)
    print("TEST 1: Normal case with cooccurrences")
    print("="*80)
    
    # Граф:
    # 0 -> {1: 10, 2: 5} (чаще 1)
    # 1 -> {0: 10}       (только 0)
    # 2 -> {}            (нет соседей)
    cooccur_counts = {
        0: {1: 10, 2: 5},
        1: {0: 10},
        2: {},
    }
    
    # Эмбеддинги (one-hot для наглядности)
    # 0: [1, 0, 0]
    # 1: [0, 1, 0]
    # 2: [0, 0, 1]
    item_embeddings = {
        0: torch.tensor([1.0, 0.0, 0.0]),
        1: torch.tensor([0.0, 1.0, 0.0]),
        2: torch.tensor([0.0, 0.0, 1.0]),
    }
    
    all_item_ids = [0, 1, 2]

    # Инициализация
    transform = AddWeightedCooccurrenceEmbeddingsVectorized(
        cooccur_counts=cooccur_counts,
        item_id_to_embedding=item_embeddings,
        all_item_ids=all_item_ids,
        device=device,
        max_neighbors=2, # Ограничим до 2 для проверки обрезки
        seed=42
    )

    # ----------------------------------------------------------------
    # Проверка 1: Айтем 1 должен всегда выбирать соседа 0 (вероятность 1.0)
    # ----------------------------------------------------------------
    batch_1 = {'item_id': torch.tensor([1, 1, 1], device=device)}
    res_1 = transform(batch_1)
    # Ожидаем эмбеддинг айтема 0: [1.0, 0.0, 0.0]
    expected_emb_0 = item_embeddings[0].to(device)
    
    print("Checking deterministic neighbor (1 -> 0)...")
    for emb in res_1['cooccurrence_embedding']:
        assert torch.allclose(emb, expected_emb_0), \
            f"Item 1 should strictly link to 0. Got {emb}"
    print("✅ Deterministic neighbor check passed")

    # ----------------------------------------------------------------
    # Проверка 2: Айтем 2 (без соседей) должен выдавать валидный случайный эмбеддинг
    # ----------------------------------------------------------------
    batch_2 = {'item_id': torch.tensor([2] * 100, device=device)}
    res_2 = transform(batch_2)
    embs_2 = res_2['cooccurrence_embedding']
    
    print("Checking fallback for item without neighbors...")
    # Проверяем, что нет NaN
    assert not torch.isnan(embs_2).any(), "NaN found in fallback embeddings"
    
    # Проверяем, что возвращаются реальные эмбеддинги из словаря
    valid_embs_set = {tuple(e.tolist()) for e in item_embeddings.values()}
    for emb in embs_2[:10]: # Проверим первые 10
        assert tuple(emb.tolist()) in valid_embs_set, f"Invalid embedding generated: {emb}"
    print("✅ Fallback check passed")

    # ----------------------------------------------------------------
    # Проверка 3: Распределение вероятностей (Item 0 -> 1(66%) vs 2(33%))
    # ----------------------------------------------------------------
    # Запустим большой батч для статистики
    batch_0 = {'item_id': torch.tensor([0] * 1000, device=device)}
    res_0 = transform(batch_0)
    embs_0 = res_0['cooccurrence_embedding']
    
    # Считаем, сколько раз выпал эмбеддинг 1 (сосед с весом 10) и эмбеддинг 2 (сосед с весом 5)
    # Emb 1 = [0, 1, 0], Emb 2 = [0, 0, 1]
    count_1 = (embs_0[:, 1] == 1.0).sum().item()
    count_2 = (embs_0[:, 2] == 1.0).sum().item()
    
    ratio = count_1 / (count_1 + count_2)
    expected_ratio = 10 / 15 # ~0.666
    
    print(f"Checking distribution for Item 0. Expected ~{expected_ratio:.2f}, Got {ratio:.2f}")
    assert abs(ratio - expected_ratio) < 0.05, \
        f"Distribution mismatch! Expected {expected_ratio:.2f}, got {ratio:.2f}"
    print("✅ Distribution check passed")

    print("\n" + "="*80)
    print("TEST 2: Edge Cases")
    print("="*80)

    # ----------------------------------------------------------------
    # Проверка 4: Пустой батч
    # ----------------------------------------------------------------
    batch_empty = {'item_id': torch.tensor([], dtype=torch.long, device=device)}
    res_empty = transform(batch_empty)
    assert res_empty['cooccurrence_embedding'].shape[0] == 0
    print("✅ Empty batch passed")

    # ----------------------------------------------------------------
    # Проверка 5: Item ID вне списка all_item_ids (например, padding index или новый айтем)
    # В текущей реализации searchsorted, индексы клемпятся.
    # Проверим, что код не падает.
    # ----------------------------------------------------------------
    unknown_id = 999
    batch_unknown = {'item_id': torch.tensor([unknown_id], device=device)}
    
    try:
        res_unknown = transform(batch_unknown)
        print("✅ Unknown item ID handled (no crash)")
        # В идеале тут надо проверить, что вернулось (скорее всего, neighbor для последнего айтема)
    except Exception as e:
        pytest.fail(f"Crashed on unknown item ID: {e}")

    # ----------------------------------------------------------------
    # Проверка 6: Воспроизводимость (Seed)
    # ----------------------------------------------------------------
    batch_seed = {'item_id': torch.tensor([0, 2, 0, 1] * 10, device=device)}
    transform_a = AddWeightedCooccurrenceEmbeddingsVectorized(
        cooccur_counts, item_embeddings, all_item_ids, device, seed=42
    )
    res_a = transform_a(batch_seed)['cooccurrence_embedding']
    transform_b = AddWeightedCooccurrenceEmbeddingsVectorized(
        cooccur_counts, item_embeddings, all_item_ids, device, seed=42
    )
    res_b = transform_b(batch_seed)['cooccurrence_embedding']

    assert torch.allclose(res_a, res_b), "Results differ with same seed!"
    print("✅ Seeding reproducibility passed")
    
    print("\n✅ ALL TESTS PASSED!")

if __name__ == "__main__":
    test_add_weighted_cooccurrence_embeddings()
