import torch

from models import RqVaeModel
from utils import DEVICE


class SimplifiedTree:
    def __init__(self, rqvae_model: RqVaeModel, device: torch.device = DEVICE):
        """
        :param rqvae_model: обученная модель rq-vae
        :param device: устройство
        """
        self.device: torch.device = device
        self.embedding_table: torch.Tensor = torch.stack(
            [cb for cb in rqvae_model.codebooks]
        ).to(self.device)  # (semantic_id_len, codebook_size, emb_dim
        self.sem_id_len, self.codebook_size, self.emb_dim = self.embedding_table.shape
        self.sem_ids_count: int = 0
        self.full_embeddings: torch.Tensor = torch.empty((0, 0))
        self.item_ids: torch.Tensor = torch.empty((0, 0))

    def build_tree_structure(self, semantic_ids: torch.Tensor, residuals: torch.Tensor, item_ids: torch.Tensor,
                             sum_with_residuals: bool = True) -> None:
        """
        :param sum_with_residuals: флаг, отвечающий за то учитывать ли остатки при выборе кандидатов
        :param semantic_ids: (sem_ids_count, sem_id_len)
        :param residuals: (sem_ids_count, emb_dim)
        :param item_ids: (sem_ids_count,)
        """
        self.sem_ids_count = semantic_ids.shape[0]
        assert residuals.shape == (self.sem_ids_count, self.emb_dim)
        assert semantic_ids.shape == (self.sem_ids_count, self.sem_id_len)
        assert item_ids.shape == (self.sem_ids_count,)

        semantic_ids = semantic_ids.to(self.device)
        residuals = residuals.to(self.device).float() if sum_with_residuals else torch.zeros_like(residuals,
                                                                                                  device=self.device,
                                                                                                  dtype=torch.float)
        self.full_embeddings = self.calculate_full(semantic_ids).float() + residuals
        self.item_ids = item_ids

    def calculate_full(self, sem_ids: torch.Tensor) -> torch.Tensor:
        """
        :param sem_ids: набор из sem ids (count, sem_id_len)
        :return: эмбеддинг для каждого sem_id из набора (count, emb_dim)
        """
        assert sem_ids.shape[1] == self.sem_id_len
        sem_ids = sem_ids.to(self.device)

        expanded_emb_table = (self.embedding_table.unsqueeze(0)
                              .expand(sem_ids.shape[0], -1, -1, -1))  # (count, sem_id_len, codebook_size, emb_dim)

        index = (sem_ids.unsqueeze(-1)
                 .expand(-1, -1, self.emb_dim)
                 .unsqueeze(2))  # (count, sem_id_len, 1, emb_dim)

        return torch.gather(input=expanded_emb_table, index=index, dim=2).sum(1).squeeze(1)  # (count, emb_dim)

    def query(self, request_sem_ids: torch.Tensor, items_to_query: int) -> torch.Tensor:
        """
        :param request_sem_ids: батч sem ids (batch_size, sem_id_len)
        :param items_to_query: количество ближайших элементов которые нужно взять (int)
        :return: тензор индексов ближайших k элементов из всех semantic_ids для каждого sem_id из батча (batch_size, k)
        """
        assert request_sem_ids.shape[1] == self.sem_id_len
        assert 0 < items_to_query <= self.sem_ids_count

        request_sem_ids = request_sem_ids.to(self.device)
        request_embeddings = self.calculate_full(request_sem_ids)  # (batch_size, emb_dim)

        request_embeddings = (request_embeddings.unsqueeze(1)
                              .expand(-1, self.sem_ids_count, -1))  # (batch_size, sem_ids_count, emb_dim)

        diff_norm = torch.norm(self.full_embeddings - request_embeddings, p=2, dim=2)  # (batch_size, sem_ids_count)

        indices = torch.argsort(diff_norm, descending=False, dim=1)[:, :items_to_query]  # (batch_size, k)
        return self.item_ids[indices]

    def _query(self, request_sem_ids: torch.Tensor, k: int) -> torch.Tensor:
        """
        Альтернатива get_ids, попытка ускорить
        :param request_sem_ids: батч sem ids (batch_size, sem_id_len)
        :param k: количество ближайших элементов которые нужно взять (int)
        :return: тензор индексов ближайших k элементов из всех semantic_ids для каждого sem_id из батча (batch_size, k)
        """
        assert request_sem_ids.shape[1] == self.sem_id_len
        assert 0 < k <= self.sem_ids_count
        request_sem_ids = request_sem_ids.to(self.device)

        index = (request_sem_ids.unsqueeze(-1)
                 .expand(-1, -1, self.emb_dim)
                 .unsqueeze(2))  # (batch_size, sem_id_len, 1, emb_dim)

        request_embeddings = torch.gather(
            input=self.embedding_table.unsqueeze(0).expand(request_sem_ids.shape[0], -1, -1, -1),
            dim=2,
            index=index
        ).sum(1)  # (batch_size, emb_dim)

        diff_norm = torch.cdist(self.full_embeddings, request_embeddings.unsqueeze(1), p=2).squeeze(
            1)  # (batch_size, sem_ids_count)

        _, indices = torch.topk(diff_norm, k=k, dim=1, largest=False)  # (batch_size, k)
        return self.item_ids[indices.squeeze(-1)]
