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
        self._sem_ids_sparse_tensor = None #тензор группирирующий остатки по semantic_id
        self.item_ids_sparse_tensor = None #тензор группирируюшщий реальные айди айтемов по semantic_id
        self.counts_dict = defaultdict(int) #тензор храняющий количество коллизий по semantic_id
        self.residual_dim = residual_dim #длина остатка
        self.emb_dim = emb_dim #длина semantic_id
        self.codebook_size = codebook_size #количество элементов в одном кодбуке
        self.device = device #девайс
        self.item_ids_dict = {} #словарь сопостовляющий каждому item_id его semantic_id и токен решающий коллизии

        self.key = torch.tensor([self.codebook_size ** i for i in range(self.emb_dim)], dtype=torch.long, device=self.device) #ключ для сопоставления числа каждому semantic_id

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Перенос тензора на устройство
        """
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        return tensor

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

        item_ids = self._to_device(item_ids)
        residuals = self._to_device(residuals)
        semantic_ids = self._to_device(semantic_ids)

        unique_id = torch.einsum('nc,c->n', semantic_ids, self.key)
        unique_ids, inverse_indices = torch.unique(unique_id, return_inverse=True)
        sorted_indices = torch.argsort(inverse_indices)
        counts = torch.bincount(inverse_indices)
        max_residuals_count = counts.max().item()
        offsets = torch.cumsum(torch.cat((torch.tensor([0], dtype=torch.long, device=self.device), counts[:-1])), dim=0)
        row_indices = inverse_indices[sorted_indices]
        col_indices = torch.arange(semantic_ids_count) - offsets[row_indices]
        indices = torch.stack([
            unique_ids[row_indices],
            col_indices
        ], dim=0)

        self._sem_ids_sparse_tensor = torch.sparse_coo_tensor(indices, residuals[sorted_indices], size=(self.codebook_size ** self.emb_dim, max_residuals_count, self.residual_dim), device=self.device)
        self.counts_dict = defaultdict(int, zip(unique_ids.tolist(), counts.tolist()))

        item_id_indices = torch.stack((unique_ids[row_indices], col_indices))

        self.item_ids_dict = {
            item_id.item(): (sem_id_key.item(), dedup_token.item())
            for item_id, (sem_id_key, dedup_token) in zip(item_ids[sorted_indices], torch.stack((unique_ids[row_indices], col_indices), dim=1))
        }
        self.item_ids_sparse_tensor = torch.sparse_coo_tensor(item_id_indices, item_ids[sorted_indices], size=(self.codebook_size ** self.emb_dim, max_residuals_count), device=self.device, dtype=torch.int16)

    def get_residuals_by_semantic_id_batch(self, semantic_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert semantic_ids.shape[1] == self.emb_dim

        semantic_ids = self._to_device(semantic_ids)
        unique_ids = torch.einsum('nc,c->n', semantic_ids, self.key)

        candidates = torch.stack([self._sem_ids_sparse_tensor[key].to_dense() for key in unique_ids])
        counts = torch.tensor([self.counts_dict[key.item()] for key in unique_ids], device=self.device)
        mask = torch.arange(candidates.shape[1], device=self.device).expand(len(unique_ids), -1) < counts.view(-1, 1)

        return candidates, mask

    def get_scores_batch(self, semantic_ids: torch.Tensor, residuals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param semantic_id: [batch_size, emb_dim] semantic ids (без токена решающего коллизии)
        :param residuals: [batch_size, residual_dim] остатки

        :return: Словарь с ключами:
            - 'scores_mask': [batch_size, max_collision_count] маска существующих значений scores
            - 'scores': [batch_size, max_collision_count] софтмакс для каждого из кандидатов
            - 'dedup_tokens_mask': [batch_size] маска существующих токенов решающих коллизии
            - 'dedup_tokens': [batch_size] токены решающие коллизии
            - 'item_ids': [batch_size] реальные айди айтемов
        """
        assert semantic_ids.shape[1] == self.emb_dim
        assert residuals.shape[1] == self.residual_dim
        assert semantic_ids.shape[0] == residuals.shape[0]

        semantic_ids = self._to_device(semantic_ids)
        residuals = self._to_device(residuals)

        unique_ids = torch.einsum('nc,c->n', semantic_ids, self.key)

        candidates, mask = self.get_residuals_by_semantic_id_batch(semantic_ids)

        scores = torch.softmax(torch.einsum('njk,nk->nj', candidates, residuals).masked_fill(~mask, float('-inf')), dim=1)

        indices = torch.argmax(scores, dim=1)
        item_ids = torch.stack([self.item_ids_sparse_tensor[unique_ids[i]][indices[i]] for i in range(semantic_ids.shape[0])])

        return {
            "scores_mask": mask,
            "scores": scores,
            "dedup_tokens_mask": mask.any(dim=1),
            "dedup_tokens": indices,
            "item_ids": item_ids
        }


    def get_item_id_info(self, item_id: int) -> dict[str, torch.Tensor]:
        """
        Возвращает информацию по заданному item_id:
        - semantic_id
        - Все элементы с таким же semantic_id
        - Их item_ids
        - Их остатки
        - Токены, решающие коллизии (dedup tokens)

        :param item_id: айди айтема

        :return: Словарь с ключами:
            - 'semantic_id': [emb_dim] semantic id
            - 'residuals': [count, residual_dim] остатки
            - 'item_ids': [count] item ids
            - 'dedup_tokens': [count] токены решающие коллизии
        """

        if item_id not in self.item_ids_dict:
            return {
                "semantic_id": torch.empty(0, dtype=torch.long, device=self.device),
                "residuals": torch.empty((0, self.residual_dim), device=self.device),
                "item_ids": torch.empty(0, dtype=torch.int16, device=self.device),
                "dedup_tokens": torch.empty(0, dtype=torch.long, device=self.device),
            }

        semantic_id_key, dedup_token = self.item_ids_dict[item_id]

        semantic_id = torch.div(semantic_id_key, self.key, rounding_mode='floor') % self.codebook_size

        assert semantic_id.shape == (self.emb_dim,)

        candidates, mask = self.get_residuals_by_semantic_id_batch(semantic_id[None])
        residuals = candidates.squeeze(0)[mask.squeeze(0)]
        item_ids = self.item_ids_sparse_tensor[semantic_id_key].to_dense()[mask.squeeze(0)]

        dedup_tokens = torch.arange(residuals.shape[0], device=self.device)

        return {
            "semantic_id": semantic_id,
            "residuals": residuals,
            "item_ids": item_ids,
            "dedup_tokens": dedup_tokens,
        }

    def get_item_ids_batch(self, semantic_ids: torch.Tensor, dedup_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param semantic_id: [batch_size, emb_dim] semantic ids (без токенов решающего коллизии)
        :param dedup_tokens: [batch_size] токены решающие коллизии

        :return: item_ids : [batch_size] реальные айди айтемов
        """
        assert semantic_ids.shape[1] == self.emb_dim
        assert dedup_tokens.shape == (semantic_ids.shape[0],)

        semantic_ids = self._to_device(semantic_ids)
        dedup_tokens = self._to_device(dedup_tokens)

        unique_ids = torch.einsum('nc,c->n', semantic_ids, self.key)

        item_ids = torch.stack([self.item_ids_sparse_tensor[unique_ids[i]][dedup_tokens[i]] for i in range(semantic_ids.shape[0])])

        return item_ids
