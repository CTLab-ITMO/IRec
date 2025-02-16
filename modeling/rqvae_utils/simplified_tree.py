import torch


class SimplifiedTree:
    def __init__(self, embedding_table: torch.Tensor, device: torch.device = torch.device('cpu')):
        """
        :param embedding_table: Тензор из RQ-VAE # (semantic_id_len, codebook_size, emb_dim)
        :param device: Устройство
        """
        self.embedding_table: torch.Tensor = embedding_table.to(device)  # (semantic_id_len, codebook_size, emb_dim)
        self.sem_id_len, self.codebook_size, self.emb_dim = embedding_table.shape
        self.device: torch.device = device
        self.sem_ids_count: int = 0
        self.full_embeddings: torch.Tensor = torch.empty((0, 0))

    def init_tree(self, embeddings: torch.Tensor) -> None:
        """
        :param embeddings: тензор эмбеддингов для каждого из semantic ids (sem_ids_count, emb_dim)
        """
        assert embeddings.shape[1] == self.emb_dim
        self.full_embeddings = embeddings.to(self.device)  # (sem_ids_count, emb_dim)
        self.sem_ids_count = embeddings.shape[0]

    def get_ids(self, request_sem_ids: torch.Tensor, k: int) -> torch.Tensor:
        """
        :param request_sem_ids: батч из sem ids (batch_size, sem_id_len)
        :param k: количество ближайших элементов которые нужно взять (int)
        :return: тензор индексов ближайших k элементов из всех semantic_ids для каждого sem_id из батча (batch_size, k)
        """
        assert request_sem_ids.shape[1] == self.sem_id_len
        assert 0 < k <= self.sem_ids_count
        request_sem_ids = request_sem_ids.to(self.device)

        expanded_emb_table = (self.embedding_table.unsqueeze(0)
                              .expand(request_sem_ids.shape[0], -1, -1,
                                      -1))  # (batch_size, sem_id_len, codebook_size, emb_dim)

        index = (request_sem_ids.unsqueeze(-1)
                 .expand(-1, -1, self.emb_dim)
                 .unsqueeze(2))  # (batch_size, sem_id_len, 1, emb_dim)

        request_embeddings = (torch.gather(input=expanded_emb_table, index=index, dim=2).sum(1)
                              .expand(-1, self.sem_ids_count, -1))  # (batch_size, sem_ids_count, emb_dim)

        diff_norm = torch.norm(self.full_embeddings - request_embeddings, p=2, dim=2)  # (batch_size, sem_ids_count)

        indices = torch.argsort(diff_norm, descending=False, dim=1)[:, :k]  # (batch_size, k)
        return indices
