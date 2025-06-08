import torch
import torch.nn as nn


class TestTiger(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sem_id_len = 4

        self.position_embeddings = nn.Embedding(num_embeddings=200, embedding_dim=64)

        self.sem_id_position_embeddings = nn.Embedding(num_embeddings=4, embedding_dim=64)

        self.bos_embedding = nn.Parameter(torch.randn(64))

        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=256, embedding_dim=64)
            for _ in range(4)
        ])

    def _embed_semantic_tokens(self, sem_ids: torch.LongTensor) -> torch.Tensor:
        """
        sem_ids: (N,)
        embeds: (N, D)
        """

        # НЕ МЕНЯТЬ вроде иначе градиент не будет норм протекать
        W = torch.stack([emb.weight for emb in self.codebook_embeddings], dim=0)

        device = sem_ids.device
        positions = torch.arange(sem_ids.size(0), device=device)  # (N,)
        book_idx = positions % self.sem_id_len  # (N,)

        return W[book_idx, sem_ids]  # (N , D)

    def _get_position_embeddings(self, mask: torch.BoolTensor) -> torch.Tensor:
        B, L = mask.shape
        device = mask.device
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embeddings(position_ids)  # (B, max_seq_len, D)
        pos_emb[~mask] = 0.0
        return pos_emb  # (B, max_seq_len, D)

    def _get_sem_ids_position_embeddings(self, mask: torch.BoolTensor) -> torch.Tensor:
        B, L = mask.shape
        device = mask.device
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        sem_pos_ids = position_ids.remainder(self.sem_id_len)
        sem_pos_emb = self.sem_id_position_embeddings(sem_pos_ids)  # (B, max_seq_len, D)
        sem_pos_emb[~mask] = 0.0
        return sem_pos_emb

        return total_emb  # (B, max_seq_len, D)

    def prepare_sem_id_batch(
            self,
            sem_embs: torch.Tensor,  # (B, max_seq_len, D)
            lengths: torch.LongTensor  # (B,) длины в токенах
    ):
        B, L, D = sem_embs.shape
        device = sem_embs.device

        total_lens = lengths
        item_starts = total_lens - self.sem_id_len

        # Декодер
        idx = item_starts.unsqueeze(1) + torch.arange(self.sem_id_len, device=device).unsqueeze(0)  # (B, sem_id_len)
        dec_tokens = sem_embs.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))  # (B, sem_id_len, D)

        bos = self.bos_embedding.unsqueeze(0).unsqueeze(1).expand(B, 1, D)  # (B, 1, D)
        decoder_emb = torch.cat([bos, dec_tokens], dim=1)  # (B, sem_id_len + 1, D)

        sem_pos_ids = torch.arange(self.sem_id_len, device=device).unsqueeze(0).expand(B, -1)  # (B, sem_id_len)
        sem_pos_emb = self.sem_id_position_embeddings(sem_pos_ids)  # (B, sem_id_len, D)
        decoder_emb[:, 1:] += sem_pos_emb  # только к токенам после BOS

        # Энкодер
        enc_lens = lengths - self.sem_id_len  # (B,)
        max_enc_len = enc_lens.max().item()  # без BOS пока

        range_row = torch.arange(max_enc_len, device=device).unsqueeze(0)  # (1, max_enc_len)
        mask = range_row < enc_lens.unsqueeze(1)  # (B, max_enc_len)

        encoder_body = torch.zeros(B, max_enc_len, D, device=device)
        encoder_body[mask] = sem_embs[:, :max_enc_len][mask]

        pos_emb = self._get_position_embeddings(mask)
        sem_pos_emb = self._get_sem_ids_position_embeddings(mask)
        encoder_body += pos_emb + sem_pos_emb  # (B, max_enc_len, D)

        bos = self.bos_embedding.view(1, 1, -1).expand(B, 1, D)
        encoder_emb = torch.cat([bos, encoder_body], dim=1)  # (B, max_enc_len+1, D)

        bos_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        encoder_mask = torch.cat([bos_mask, mask], dim=1)  # (B, max_enc_len+1)

        return encoder_emb, encoder_mask, decoder_emb