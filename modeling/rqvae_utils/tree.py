import numpy as np
import torch

from utils import DEVICE


class Tree:
    def __init__(self, embedding_table, device: torch.device = DEVICE):
        """
        :param embedding_table: Тензор из RQ-VAE # (semantic_id_len, codebook_size, emb_dim)
        :param device: Устройство
        """
        self.embedding_table: torch.Tensor = embedding_table  # (semantic_id_len, codebook_size, emb_dim)
        self.sem_id_len, self.codebook_size, self.emb_dim = embedding_table.shape
        self.device: torch.device = device
        self.key: torch.Tensor = torch.empty((0, 0))
        self.A: torch.Tensor = torch.empty((0, 0))  # будет (max_sem_id, )
        self.sem_ids_count: int = -1
        self.sem_ids_embs: torch.Tensor = torch.empty((0, 0))
        self.sids: torch.Tensor = torch.empty((0, 0))  # будет (sem_id_len, )

    def init_tree(self, semantic_ids, residuals):
        """
        :param semantic_ids: (sem_ids_count, sem_id_len)
        :param residuals: (sem_ids_count, emb_dim)
        """

        assert semantic_ids.shape[0] == residuals.shape[0]
        assert semantic_ids.shape[1] == self.sem_id_len
        assert residuals.shape[1] == self.emb_dim

        self.sem_ids_count = semantic_ids.shape[0]
        self.key = torch.tensor([self.codebook_size ** i for i in range(self.sem_id_len - 1, -1, -1)],
                                dtype=torch.long, device=self.device)
        self.sids = self.get_sids(semantic_ids.float())  # (sem_id_len, )
        self.sem_ids_embs = self.calculate_full(semantic_ids, residuals)

        result = torch.full(size=[self.codebook_size ** self.sem_id_len], fill_value=0, dtype=torch.int64,
                            device=self.device)
        temp_unique_id = self.sids * self.codebook_size
        temp_sem_ids = torch.concat([semantic_ids, torch.zeros(self.sem_ids_count, device=self.device).unsqueeze(1)],
                                    dim=-1)

        for i in range(0, self.sem_id_len + 1):
            temp_unique_id = temp_unique_id - (self.codebook_size ** i) * temp_sem_ids[:, self.sem_id_len - i]
            temp_unique_ids, temp_inverse_indices = torch.unique(temp_unique_id, return_inverse=True)
            temp_counts = torch.bincount(temp_inverse_indices)
            truncated_ids = torch.floor_divide(input=temp_unique_id, other=(self.codebook_size ** (i + 1))).long()
            result[truncated_ids] = temp_counts[temp_inverse_indices]

        self.A = result

    def get_counts(self, sem_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param sem_ids: (batch_size, sem_id_len)
        :return: префиксы всех длин sem_ids, количество sem_id на каждой глубине дерева
        """
        assert sem_ids.shape[1] == self.sem_id_len
        assert sem_ids.device == self.device

        offsets = torch.arange(self.sem_id_len + 1, device=self.device)
        i = torch.arange(self.sem_id_len, device=self.device)

        mask_sem = (i < (self.sem_id_len - offsets.unsqueeze(1))).long()  # (sem_id_len + 1, sem_id_len)
        divs = torch.pow(self.codebook_size, offsets)  # (sem_id_len + 1,)

        C = (sem_ids.unsqueeze(1) * mask_sem.unsqueeze(0) * self.key.unsqueeze(0).unsqueeze(1)).sum(dim=-1)
        B = C // divs.unsqueeze(0)

        return C, self.A[B]  # (batch_size, sem_id_len + 1), (batch_size, sem_id_len + 1)

    def get_sids(self, sem_ids: torch.Tensor) -> torch.Tensor:
        """
        :param sem_ids: (sem_id_count, sem_id_len)
        :return: хэши sem_ids (sem_id_count,)
        """
        assert sem_ids.shape[1] == self.sem_id_len
        return torch.einsum('nc,c->n', sem_ids, self.key.float())  # (sem_ids_count,)

    def calc_ol(self, batch_ids: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param batch_ids: (batch_size, sem_id_len)
        :param k: int
        :return: тензор глубин на которые нужно подняться (batch_size,), маска для sem_id для нужной глубины (batch_size, sem_ids_count)
        """
        assert batch_ids.shape[1] == self.sem_id_len
        assert k < self.sem_ids_count  # корректный сценарий когда тензор не пустой

        c, a = self.get_counts(batch_ids)
        ol = torch.argmax((a > k).long(), dim=-1)  # (bs,)
        gather_ol = torch.gather(c, dim=1, index=ol.unsqueeze(1)).squeeze()  # (bs,)

        mask_ol = (gather_ol.unsqueeze(-1) <= self.sids) & (
                self.sids < (gather_ol + torch.pow(self.codebook_size, ol)).unsqueeze(-1))
        return ol, mask_ol  # (bs,) (bs, sem_ids_count)

    def calc_il(self, batch_ids, k):
        """
        :param batch_ids: (batch_size, sem_id_len)
        :param k: int
        :return: тензор глубин на которые нужно подняться (batch_size,), маска для sem_id для нужной глубины (batch_size, sem_ids_count)
        """
        assert batch_ids.shape[1] == self.sem_id_len
        assert k < self.sem_ids_count  # корректный сценарий когда тензор не пустой

        batch_dim = batch_ids.shape[0]
        c, a = self.get_counts(batch_ids)
        extended_c = torch.concat([torch.tensor(float("inf"), device=self.device).expand(batch_dim, 1), c], dim=1)

        il = torch.argmax((a > k).long(), dim=-1) - 1  # (bs,)
        gather_il = torch.gather(extended_c, dim=1, index=(il + 1).unsqueeze(1)).squeeze()  # (bs,)

        mask_il = (gather_il.unsqueeze(-1) <= self.sids) & (
                self.sids < (gather_il + torch.pow(self.codebook_size, il)).unsqueeze(-1))
        return il, mask_il  # (bs,) (bs, sem_ids_count)

    def get_repeated_sids(self, k: int) -> torch.Tensor:
        return self.sids.repeat(k, 1)  # (k, sem_ids_count)

    def get_request_embeddings(self, decomposed_embeddings: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        """
        :param decomposed_embeddings: разложение sem_ids на эмбеддинги (count, sem_id_len +1, emb_dim)
        :param levels: сколько нужно взять эмбеддингов для суммы для каждого sem_id (count,)
        :return: эмбеддинги sem_id для нужных глубин (count, emb_dim)
        """
        assert decomposed_embeddings.shape[1:] == (self.sem_id_len + 1, self.emb_dim)
        assert levels.shape == (decomposed_embeddings.shape[0],)

        mask = torch.arange(1, self.sem_id_len + 2, device=self.device) >= torch.arange(self.sem_id_len + 2, 0, -1,
                                                                                        device=self.device).unsqueeze(1)
        return torch.sum(decomposed_embeddings * mask[levels + 1].unsqueeze(-1), dim=1)  # (bs, emb_dim)

    def calculate_full(self, sem_ids: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        """
        :param sem_ids: sem_ids (count, sem_id_len)
        :param residuals: остатки для каждого sem_id (count, emb_dim)
        :return: полные эмбеддинги для каждого айтема (count, emb_dim)
        """
        assert sem_ids.shape[1] == self.sem_id_len
        assert residuals.shape[1] == self.emb_dim
        assert residuals.shape[0] == sem_ids.shape[0]

        count = residuals.shape[0]
        index = sem_ids.view(count, -1, 1, 1).expand(-1, -1, -1, self.emb_dim)
        embs = torch.gather(input=self.embedding_table.unsqueeze(0).expand(count, -1, -1, -1), dim=2,
                            index=index)  # expand бесплатный по памяти
        decomposed_embs = torch.concat([embs.squeeze(2), residuals.unsqueeze(1)], dim=1)  # (sem_ids_count, emb_dim)

        assert decomposed_embs.shape == (sem_ids.shape[0], self.sem_id_len + 1, self.emb_dim)
        return decomposed_embs

    def calculate_level_embeddings(self, decomposed_embeddings: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        """
        :param decomposed_embeddings: разложение sem_ids на эмбеддинги (count, sem_id_len +1, emb_dim)
        :param levels: сколько нужно взять эмбеддингов для суммы для каждого sem_id (count,)
        :return: эмбеддинги для всех sem_ids для нужных глубин (batch_size, sem_ids_count, emb_dim)
        """
        assert decomposed_embeddings.shape == (self.sem_ids_count, self.sem_id_len + 1, self.emb_dim)

        mask = (torch.arange(1, self.sem_id_len + 2, device=self.device) >=
                torch.arange(self.sem_id_len + 2, 0, -1,device=self.device).unsqueeze(1)).float()
        sids_mask = mask[levels + 1].unsqueeze(-1)  # (batch_size, sem_id_len + 1, 1)
        return torch.einsum('nld,bld->bnd', decomposed_embeddings, sids_mask)  # (batch_size, sem_ids_count, emb_dim)

    def mask_result(self, result: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, result, torch.tensor(float('-inf'), device=self.device))

    def get_ids(self, request_sem_ids: torch.Tensor, request_residuals: torch.Tensor, k: int) -> torch.Tensor:
        """
        :param request_sem_ids: батч из sem_ids (batch_size, sem_id_len)
        :param request_residuals: батч из остатков (batch_size, emb_dim)
        :param k: количество ближайших элементов которые нужно взять int
        :return: тензор индексов ближайших k элементов из всех semantic_ids для каждого sem_id из батча (batch_size, k)
        """
        assert request_sem_ids.shape[0] == request_residuals.shape[0]
        assert request_sem_ids.shape[1] == self.sem_id_len
        assert request_residuals.shape[1] == self.emb_dim
        assert 0 <= k < self.sem_ids_count

        ol, ol_mask = self.calc_ol(request_sem_ids, k)
        il, il_mask = self.calc_il(request_sem_ids, k)

        il_mask = il_mask.detach().cpu()
        ol_mask = ol_mask.detach().cpu()

        ol_mask = ol_mask & ~il_mask

        request_embs = self.calculate_full(request_sem_ids, request_residuals)

        ol_sids_embeddings = self.calculate_level_embeddings(self.sem_ids_embs, ol)
        il_sids_embeddings = self.calculate_level_embeddings(self.sem_ids_embs, il)

        ol_request_embeddings = self.get_request_embeddings(request_embs, ol)
        il_request_embeddings = self.get_request_embeddings(request_embs, il)

        ol_scores = torch.matmul(ol_sids_embeddings, ol_request_embeddings.unsqueeze(-1)).squeeze(-1).detach().cpu()

        il_scores = torch.matmul(il_sids_embeddings, il_request_embeddings.unsqueeze(-1)).squeeze(-1).detach().cpu()

        ids = np.lexsort(keys=(-torch.cat([il_scores, ol_scores], dim=1),
                               ~torch.cat([torch.ones_like(il_mask), torch.zeros_like(ol_mask)], dim=1),
                               ~torch.cat([il_mask, ol_mask], dim=1)))

        return (ids % self.sem_ids_count)[:, :self.sem_ids_count][:, :k]  # (batch_size, k)
