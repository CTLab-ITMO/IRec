import torch

class CollisionSolver:
    def __init__(self, residual_dim, emb_dim, codebook_size, device: torch.device = torch.device('cpu')):
        """
        :param residual_dim: Длина остатка
        :param codebook_size: Количество элементов в одном кодбуке
        :param emb_dim: Длина semantic_id (без токена решающего коллизии)
        :param device: Устройство
        """
        self._semantic_id_dict = None
        self._unique_ids = None
        self.counts = None
        self.residual_dim = residual_dim
        self.emb_dim = emb_dim
        self.codebook_size = codebook_size
        self.device = device

        self.key = torch.tensor([self.codebook_size ** i for i in range(self.emb_dim)], dtype=torch.long, device=self.device)

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Перенос тензора на устройство
        """
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        return tensor

    def create_query_candidates_dict(self, semantic_ids: torch.Tensor, residuals: torch.Tensor) -> None:
        """
        Создает разреженный тензор, который содержит сгруппированные по semantic id элементы

        :param semantic_ids: Тензор всех semantic_id, полученных из rq-vae (без токенов решающих коллизии)
        :param residuals: Тензор остатков для каждого semantic_id
        """
        residuals_count, residual_length = residuals.shape
        semantic_ids_count, semantic_id_length = semantic_ids.shape

        assert residuals_count == semantic_ids_count
        assert semantic_id_length == self.emb_dim
        assert residual_length == self.residual_dim

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
            row_indices,
            col_indices
        ], dim=0)

        self._semantic_id_dict = torch.sparse_coo_tensor(indices, residuals[sorted_indices], size=(len(unique_ids), max_residuals_count, self.residual_dim), device=self.device)
        self._unique_ids = unique_ids
        self.counts = counts


    def get_residuals_by_semantic_id(self, semantic_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param semantic_id: [emb_dim] semantic id (без токена решающего коллизии)

        :return: candidates: [max_residuals_count, residual_dim] список остатков с таким же semantic id
        :return: mask: [max_residuals_count] маска для списка остатков
        """
        assert semantic_id.shape == (self.emb_dim,)

        semantic_id = self._to_device(semantic_id)

        unique_id = torch.einsum('c,c->', semantic_id, self.key)
        target_index = (self._unique_ids == unique_id).nonzero(as_tuple=True)[0]

        if len(target_index) == 0:
            empty_candidates = torch.empty((0, self.residual_dim), device=self.device)
            empty_mask = torch.empty((0,), dtype=torch.bool, device=self.device)
            return empty_candidates, empty_mask # Если unique_id отсутствует

        candidates = self._semantic_id_dict[target_index.item()].to_dense()
        mask = torch.arange(candidates.size(0)) < self.counts[target_index.item()]

        return candidates, mask

    def get_scores(self, semantic_id, residual) -> tuple[torch.Tensor, int]:
        """
        :param semantic_id: [emb_dim] semantic id (без токена решающего коллизии
        :param residual: [residual_dim] Остаток

        :return: scores: [residuals_count] Вероятности для остатков
        :return: index: Индекс наибольшего значения в scores
        """
        assert semantic_id.shape == (self.emb_dim,)
        assert residual.shape == (self.residual_dim,)

        residual = self._to_device(residual)
        candidates, mask = self.get_residuals_by_semantic_id(semantic_id)
        scores = torch.softmax(torch.einsum('jk,k->j', candidates[mask], residual), dim=0)
        if scores.shape[0] == 0:
            return torch.empty((0,), device=self.device), torch.empty((0,), device=self.device)
        return scores, torch.argmax(scores)