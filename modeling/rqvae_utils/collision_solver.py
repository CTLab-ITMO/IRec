from collections import defaultdict

import torch

from utils import DEVICE


class CollisionSolver:
    def __init__(self,
                 emb_dim: int,
                 sem_id_len: int,
                 codebook_size: int,
                 device: torch.device=DEVICE):
        """
        :param emb_dim: Длина остатка
        :param codebook_size: Количество элементов в одном кодбуке
        :param sem_id_len: Длина semantic_id (без токена решающего коллизии)
        :param device: Устройство
        """
        self._sem_ids_sparse_tensor: torch.Tensor = torch.empty((0, 0))  # тензор группирирующий остатки по semantic_id
        self.item_ids_sparse_tensor: torch.Tensor = torch.empty(
            (0, 0))  # тензор группирирующий реальные айди айтемов по semantic_id
        self.counts_dict: dict[int, int] = defaultdict(int)  # тензор храняющий количество коллизий по semantic_id
        self.emb_dim: int = emb_dim  # длина остатка
        self.sem_id_len: int = sem_id_len  # длина semantic_id
        self.codebook_size: int = codebook_size  # количество элементов в одном кодбуке
        self.device: torch.device = device  # девайс

        self.key: torch.Tensor = torch.tensor([self.codebook_size ** i for i in range(self.sem_id_len)],
                                              dtype=torch.long,
                                              device=self.device)  # ключ для сопоставления числа каждому semantic_id

    def create_query_candidates_dict(self, item_ids: torch.Tensor, semantic_ids: torch.Tensor,
                                     residuals: torch.Tensor) -> None:
        """
        Создает разреженный тензор, который содержит сгруппированные по semantic id элементы

        :param item_ids: Реальные айди айтемов (пусть будут больше 0) (count,)
        :param semantic_ids: Тензор всех semantic_id, полученных из rq-vae (без токенов решающих коллизии) (count, sem_id_len)
        :param residuals: Тензор остатков для каждого semantic_id (count, emb_dim)
        """
        residuals_count, residual_length = residuals.shape
        semantic_ids_count, semantic_id_length = semantic_ids.shape

        assert residuals_count == semantic_ids_count
        assert semantic_id_length == self.sem_id_len
        assert residual_length == self.emb_dim
        assert item_ids.shape == (residuals_count,)

        item_ids = item_ids.to(self.device)
        residuals = residuals.to(self.device)
        semantic_ids = semantic_ids.to(self.device)

        unique_id = (semantic_ids * self.key).sum(dim=1)  # хэши
        unique_ids, inverse_indices, counts = torch.unique(unique_id, return_inverse=True, return_counts=True)
        sorted_indices = torch.argsort(inverse_indices)  # сортированные индексы чтобы совпадающие хэши шли подряд

        row_indices = inverse_indices[sorted_indices]  # отсортированные хэши

        offsets = torch.cumsum(counts, dim=0) - counts
        col_indices = torch.arange(semantic_ids_count, device=self.device) - offsets[
            row_indices]  # индексы от 0 до k внутри каждого набора из совпадающих хэшей

        indices = torch.stack([
            unique_ids[row_indices],
            col_indices
        ],
            dim=0)  # индексы для разреженного тензора: 1 размерность хэш, 2 размерность индексы от 0 до k для коллизий каждого хэша

        max_residuals_count = int(counts.max().item())  # максимальное количество коллизий для одного sem_id
        max_sid = int(self.codebook_size ** self.sem_id_len)  # максимальный хэш sem_id который может быть

        self._sem_ids_sparse_tensor = torch.sparse_coo_tensor(indices, residuals[sorted_indices],
                                                              size=(max_sid, max_residuals_count, self.emb_dim),
                                                              device=self.device)  # (max_sid, max_residuals_count, emb_dim)

        self.counts_dict = defaultdict(int, zip(unique_ids.tolist(), counts.tolist()))  # sid -> collision count

        self.item_ids_sparse_tensor = torch.sparse_coo_tensor(indices, item_ids[sorted_indices],
                                                              size=(max_sid, max_residuals_count), device=self.device,
                                                              dtype=torch.int32)  # (max_sid, max_residuals_count)

    def get_residuals_by_semantic_id_batch(self, semantic_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param semantic_ids батч из semantic ids (batch_size, sem_id_len)

        :return:
            Возвращает тензор эмбеддингов для батча semantic_ids, размерность (batch_size, max_residuals_count, emb_dim)
            Возвращает маску для этого тензора, размерность (batch_size, max_residuals_count, emb_dim)
        """
        assert semantic_ids.shape[1] == self.sem_id_len

        semantic_ids = semantic_ids.to(self.device)
        unique_ids = (semantic_ids * self.key).sum(dim=1)

        candidates = torch.stack([self._sem_ids_sparse_tensor[key].to_dense() for key in unique_ids])
        counts = torch.tensor([self.counts_dict[key.item()] for key in unique_ids], device=self.device)
        mask = torch.arange(candidates.shape[1], device=self.device).expand(len(unique_ids), -1) < counts.view(-1, 1)

        return candidates, mask

    def get_pred_scores(self, semantic_ids: torch.Tensor, pred_residuals: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        :param semantic_id: [batch_size, sem_id_len] semantic ids (без токена решающего коллизии)
        :param pred_residuals: [batch_size, emb_dim] предсказанные остатки

        :return: Словарь с ключами:
            - 'pred_scores_mask': [batch_size, max_collision_count] маска существующих значений scores для предсказанных остатков
            - 'pred_scores': [batch_size, max_collision_count] софтмакс для каждого из кандидатов для предсказанных остатков
            - 'pred_item_ids': [batch_size] реальные айди айтемов для предсказанных остатков
        """
        assert semantic_ids.shape[1] == self.sem_id_len
        assert pred_residuals.shape[1] == self.emb_dim
        assert semantic_ids.shape[0] == pred_residuals.shape[0]

        semantic_ids = semantic_ids.to(self.device)
        pred_residuals = pred_residuals.to(self.device)

        unique_ids = (semantic_ids * self.key).sum(dim=1)

        candidates, mask = self.get_residuals_by_semantic_id_batch(semantic_ids)

        pred_scores = torch.einsum('njk,nk->nj', candidates, pred_residuals).masked_fill(~mask, -torch.inf)
        pred_indices = torch.argmax(pred_scores, dim=1)
        pred_item_ids = torch.stack(
            [self.item_ids_sparse_tensor[unique_ids[i]][pred_indices[i]] for i in range(semantic_ids.shape[0])])

        return {
            "pred_scores_mask": mask,
            "pred_scores": pred_scores,
            "pred_item_ids": pred_item_ids
        }

    def get_true_dedup_tokens(self, semantic_ids: torch.Tensor, true_residuals: torch.Tensor) -> dict[
        str, torch.Tensor]:
        """
        :param semantic_id: [batch_size, sem_id_len] semantic ids (без токена решающего коллизии)
        :param true_residuals: [batch_size, emb_dim] реальные остатки

        :return: Словарь с ключами:
            - 'true_dedup_tokens': [batch_size] токены решающие коллизии для реальных остатков
        """
        assert semantic_ids.shape[1] == self.sem_id_len
        assert true_residuals.shape[1] == self.emb_dim
        assert semantic_ids.shape[0] == true_residuals.shape[0]

        semantic_ids = semantic_ids.to(self.device)
        true_residuals = true_residuals.to(self.device)

        candidates, _ = self.get_residuals_by_semantic_id_batch(semantic_ids)

        matches = torch.all(candidates == true_residuals[:, None, :], dim=2).int()
        true_dedup_tokens = torch.argmax(matches, dim=1)

        assert matches.any(dim=1).all(), "Не у всех батчей есть совпадение"

        return {
            "true_dedup_tokens": true_dedup_tokens
        }

    def get_item_ids_batch(self, semantic_ids: torch.Tensor, dedup_tokens: torch.Tensor) -> torch.Tensor:
        """
        :param semantic_id: [batch_size, sem_id_len] semantic ids (без токенов решающего коллизии)
        :param dedup_tokens: [batch_size] токены решающие коллизии

        :return: item_ids : [batch_size] реальные айди айтемов
        """
        assert semantic_ids.shape[1] == self.sem_id_len
        assert dedup_tokens.shape == (semantic_ids.shape[0],)

        semantic_ids = semantic_ids.to(self.device)
        dedup_tokens = dedup_tokens.to(self.device)

        unique_ids = (semantic_ids * self.key).sum(dim=1)

        item_ids = torch.stack(
            [self.item_ids_sparse_tensor[unique_ids[i]][dedup_tokens[i]] for i in range(semantic_ids.shape[0])])

        return item_ids
