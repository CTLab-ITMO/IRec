from collections import defaultdict

import torch


class CollisionSolver:
    def __init__(self, residual_dim, emb_dim, codebook_size, device: torch.device = torch.device('cpu')):
        """
        :param residual_dim: Длина остатка
        :param codebook_size: Количество элементов в одном кодбуке
        :param emb_dim: Длина semantic_id (без токена решающего коллизии)
        :param device: Устройство
        """
        self._sem_ids_sparse_tensor: torch.Tensor = torch.empty((0, 0)) #тензор группирирующий остатки по semantic_id
        self.item_ids_sparse_tensor: torch.Tensor = torch.empty((0, 0)) #тензор группирируюшщий реальные айди айтемов по semantic_id
        self.counts_dict: dict[int, int] = defaultdict(int) #тензор храняющий количество коллизий по semantic_id
        self.residual_dim: int = residual_dim #длина остатка
        self.emb_dim: int = emb_dim #длина semantic_id
        self.codebook_size: int = codebook_size #количество элементов в одном кодбуке
        self.device: torch.device = device #девайс

        self.key: torch.Tensor = torch.tensor([self.codebook_size ** i for i in range(self.emb_dim)], dtype=torch.long, device=self.device) #ключ для сопоставления числа каждому semantic_id

    def create_query_candidates_dict(self, item_ids: torch.Tensor, semantic_ids: torch.Tensor, residuals: torch.Tensor) -> None:
        """
        Создает разреженный тензор, который содержит сгруппированные по semantic id элементы

        :param item_ids: Реальные айди айтемов (пусть будут больше 0)
        :param semantic_ids: Тензор всех semantic_id, полученных из rq-vae (без токенов решающих коллизии)
        :param residuals: Тензор остатков для каждого semantic_id
        """
        residuals_count, residual_length = residuals.shape
        semantic_ids_count, semantic_id_length = semantic_ids.shape

        assert residuals_count == semantic_ids_count
        assert semantic_id_length == self.emb_dim
        assert residual_length == self.residual_dim
        assert item_ids.shape == (residuals_count,)

        item_ids = item_ids.to(self.device)
        residuals = residuals.to(self.device)
        semantic_ids = semantic_ids.to(self.device)

        unique_id = torch.einsum('nc,c->n', semantic_ids, self.key)
        unique_ids, inverse_indices = torch.unique(unique_id, return_inverse=True)
        sorted_indices = torch.argsort(inverse_indices)
        counts = torch.bincount(inverse_indices)
        max_residuals_count = int(counts.max().item())
        max_sid = int(self.codebook_size ** self.emb_dim)
        offsets = torch.cumsum(torch.cat((torch.tensor([0], dtype=torch.long, device=self.device), counts[:-1])), dim=0)
        row_indices = inverse_indices[sorted_indices]
        col_indices = torch.arange(semantic_ids_count) - offsets[row_indices]
        indices = torch.stack([
            unique_ids[row_indices],
            col_indices
        ], dim=0)

        self._sem_ids_sparse_tensor = torch.sparse_coo_tensor(indices, residuals[sorted_indices], size=(max_sid, max_residuals_count, self.residual_dim), device=self.device)
        self.counts_dict = defaultdict(int, zip(unique_ids.tolist(), counts.tolist()))

        item_id_indices: torch.Tensor = torch.stack((unique_ids[row_indices], col_indices))

        self.item_ids_sparse_tensor = torch.sparse_coo_tensor(item_id_indices, item_ids[sorted_indices], size=(max_sid, max_residuals_count), device=self.device, dtype=torch.int16)

    def get_residuals_by_semantic_id_batch(self, semantic_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert semantic_ids.shape[1] == self.emb_dim

        semantic_ids = semantic_ids.to(self.device)
        unique_ids = torch.einsum('nc,c->n', semantic_ids, self.key)

        candidates = torch.stack([self._sem_ids_sparse_tensor[key].to_dense() for key in unique_ids])
        counts = torch.tensor([self.counts_dict[key.item()] for key in unique_ids], device=self.device)
        mask = torch.arange(candidates.shape[1], device=self.device).expand(len(unique_ids), -1) < counts.view(-1, 1)

        return candidates, mask

    def get_pred_scores(self, semantic_ids: torch.Tensor, pred_residuals: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        :param semantic_id: [batch_size, emb_dim] semantic ids (без токена решающего коллизии)
        :param pred_residuals: [batch_size, residual_dim] предсказанные остатки

        :return: Словарь с ключами:
            - 'pred_scores_mask': [batch_size, max_collision_count] маска существующих значений scores для предсказанных остатков
            - 'pred_scores': [batch_size, max_collision_count] софтмакс для каждого из кандидатов для предсказанных остатков
            - 'pred_item_ids': [batch_size] реальные айди айтемов для предсказанных остатков
        """
        assert semantic_ids.shape[1] == self.emb_dim
        assert pred_residuals.shape[1] == self.residual_dim
        assert semantic_ids.shape[0] == pred_residuals.shape[0]

        semantic_ids = semantic_ids.to(self.device)
        pred_residuals = pred_residuals.to(self.device)

        unique_ids = torch.einsum('nc,c->n', semantic_ids, self.key)

        candidates, mask = self.get_residuals_by_semantic_id_batch(semantic_ids)

        pred_scores = torch.einsum('njk,nk->nj', candidates, pred_residuals).masked_fill(~mask, -torch.inf)
        pred_indices = torch.argmax(pred_scores, dim=1)
        pred_item_ids = torch.stack([self.item_ids_sparse_tensor[unique_ids[i]][pred_indices[i]] for i in range(semantic_ids.shape[0])])

        return {
            "pred_scores_mask": mask,
            "pred_scores": pred_scores,
            "pred_item_ids": pred_item_ids
        }

    def get_true_dedup_tokens(self, semantic_ids: torch.Tensor, true_residuals: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        :param semantic_id: [batch_size, emb_dim] semantic ids (без токена решающего коллизии)
        :param true_residuals: [batch_size, residual_dim] реальные остатки

        :return: Словарь с ключами:
            - 'true_dedup_tokens': [batch_size] токены решающие коллизии для реальных остатков
        """
        assert semantic_ids.shape[1] == self.emb_dim
        assert true_residuals.shape[1] == self.residual_dim
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
