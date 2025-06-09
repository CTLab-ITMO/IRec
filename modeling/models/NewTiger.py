import torch
import torch.nn as nn

from utils import get_activation_function, create_masked_tensor, DEVICE


class NewTiger(nn.Module):
    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            num_items,
            embedding_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout=0.0,
            activation="relu",
            layer_norm_eps=1e-9,
            initializer_range=0.02,
    ):
        super().__init__()

        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix

        self._num_items = num_items
        self._num_heads = num_heads

        self._embedding_dim = embedding_dim
        self._sem_id_len = 4

        self.position_embeddings = nn.Embedding(num_embeddings=200, embedding_dim=self._embedding_dim)

        self.sem_id_position_embeddings = nn.Embedding(num_embeddings=self._sem_id_len,
                                                       embedding_dim=self._embedding_dim)

        self.bos_embedding = nn.Parameter(torch.randn(64))

        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=256, embedding_dim=self._embedding_dim)
            for _ in range(4)
        ])

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self._embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )

        self._decoder = nn.TransformerDecoder(
            transformer_decoder_layer, num_decoder_layers
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_encoder_layers)

        self._layernorm = nn.LayerNorm(self._embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        self._init_weights(initializer_range)

    def _embed_semantic_tokens(self, sem_ids: torch.LongTensor) -> torch.Tensor:
        """
        sem_ids: (N,)
        embeds: (N, embedding_dim)
        """

        # НЕ МЕНЯТЬ вроде иначе градиент не будет норм протекать
        W = torch.stack([emb.weight for emb in self.codebook_embeddings], dim=0)
        positions = torch.arange(sem_ids.size(0), device=DEVICE)  # (N,)
        book_idx = positions % self._sem_id_len  # (N,)

        return W[book_idx, sem_ids]  # (N , embedding_dim)

    def _get_position_embeddings(self, mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embeddings(position_ids)  # (batch_size, max_seq_len, embedding_dim)
        pos_emb[~mask] = 0.0
        return pos_emb  # (batch_size, max_seq_len, embedding_dim)

    def _get_sem_ids_position_embeddings(self, mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch_size, -1)
        sem_pos_ids = position_ids.remainder(self._sem_id_len)
        sem_pos_emb = self.sem_id_position_embeddings(sem_pos_ids)  # (batch_size, max_seq_len, embedding_dim)
        sem_pos_emb[~mask] = 0.0
        return sem_pos_emb  # (batch_size, max_seq_len, embedding_dim)

    def prepare_sem_id_batch(
            self,
            sem_embs: torch.Tensor,  # (batch_size, max_seq_len, embedding_dim)
            lengths: torch.LongTensor  # (batch_size,) длины в токенах sem id
    ):
        batch_size, max_seq_len, emb_dimm = sem_embs.shape

        total_lens = lengths
        item_starts = total_lens - self._sem_id_len

        # Декодер
        idx = (item_starts.unsqueeze(1) +
               torch.arange(self._sem_id_len, device=DEVICE).unsqueeze(0))  # (batch_size, sem_id_len)
        decoder_target_embs = sem_embs.gather(1, idx.unsqueeze(-1)
                                              .expand(-1, -1, emb_dimm))  # (batch_size, sem_id_len, embedding_dim)

        bos = self.bos_embedding.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, emb_dimm)  # (B, 1, D)
        decoder_embs = torch.cat([bos, decoder_target_embs], dim=1)  # (batch_size, sem_id_len + 1, D)

        sem_pos_ids = (torch.arange(self._sem_id_len, device=DEVICE)
                       .unsqueeze(0)
                       .expand(batch_size, -1))  # (batch_size, sem_id_len)
        sem_pos_emb = self.sem_id_position_embeddings(sem_pos_ids)  # (batch_size, sem_id_len, D)
        decoder_embs[:, 1:] += sem_pos_emb  # только к токенам после BOS

        # Энкодер
        enc_lens = lengths - self._sem_id_len  # (batch_size,)
        max_enc_len = enc_lens.max().item()  # без BOS пока

        range_row = torch.arange(max_enc_len, device=DEVICE).unsqueeze(0)  # (1, max_enc_len)
        mask = range_row < enc_lens.unsqueeze(1)  # (batch_size, max_enc_len)

        encoder_body = torch.zeros(batch_size, max_enc_len, emb_dimm, device=DEVICE)
        encoder_body[mask] = sem_embs[:, :max_enc_len][mask]

        pos_emb = self._get_position_embeddings(mask)
        sem_pos_emb = self._get_sem_ids_position_embeddings(mask)
        encoder_body += pos_emb + sem_pos_emb  # (batch_size, max_enc_len, embedding_dim)

        bos = self.bos_embedding.view(1, 1, -1).expand(batch_size, 1, emb_dimm)
        encoder_embs = torch.cat([bos, encoder_body], dim=1)  # (batch_size, max_enc_len+1, embedding_dim)

        bos_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=DEVICE)
        encoder_mask = torch.cat([bos_mask, mask], dim=1)  # (batch_size, max_enc_len+1)

        return encoder_embs, encoder_mask, decoder_embs

    def forward(self, inputs):
        all_sample_events = inputs[
            "{}.ids".format(self._sequence_prefix)
        ]  # (all_batch_events)
        all_sample_lengths = inputs[
            "{}.length".format(self._sequence_prefix)
        ]  # (batch_size)

        embeddings_flat = self._embed_semantic_tokens(all_sample_events)

        assert embeddings_flat.shape[0] == sum(all_sample_lengths)

        embeddings, mask = create_masked_tensor(
            data=embeddings_flat, lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        (encoder_input_emb,  # (batch_size, seq_len - 4 + 1, embedding_dim)
         encoder_input_mask,  # (batch_size, seq_len - 4 + 1)
         decoder_input_embs) = (  # (batch_size, sem_id_len + 1, embedding_dim)
            self.prepare_sem_id_batch(embeddings, all_sample_lengths)
        )

        if self.training:
            after_encoder_emb, after_encoder_mask = self._apply_encoder(encoder_input_emb, encoder_input_mask)

            # последние sem ids
            # TODO ПРОВЕРИТЬ РАБОТОСПОСОБНОСТЬ, ЭТО ДОЛЖНО ВОЗВРАЩАТЬ ЧЕТЫРЕ ПОСЛЕДНИХ ВАЛИДНЫХ ТОКЕНА ВНУТРИ КАЖДОГО БАТЧА
            events_2d = torch.zeros_like(mask, dtype=torch.long, device=DEVICE)
            events_2d[mask] = all_sample_events
            target_tokens = events_2d[:, -self._sem_id_len:]  # (batch_size, sem_id_len)

            # Подготовка входа декодера (BOS + первые 3 токена)
            tgt = decoder_input_embs[:, :-1, :]  # (batch_size, sem_id_len, embedding_dim)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1), device=DEVICE
            )  # (sem_id_len, sem_id_len)

            decoder_output = self._decoder(
                tgt=tgt,
                memory=after_encoder_emb,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=~after_encoder_mask
                # должно быть True для паддинга и False для реальных позиций
            )  # (batch_size, sem_id_len, embedding_dim)

            losses = []  # [(batch_size, codebook_size) * sem_id_len]
            scores = []  # [(batch_size, codebook_size) * sem_id_len]
            argmaxes = []  # [(batch_size, ) * sem_id_len]

            for i in range(self._sem_id_len):
                logits = torch.matmul(
                    decoder_output[:, i, :], self.codebook_embeddings[i].weight.t()
                )  # (batch_size, codebook_size)
                scores.append(logits)

                pred_tokens = torch.argmax(logits, dim=-1)  # (batch_size,)
                argmaxes.append(pred_tokens)

                loss = nn.functional.cross_entropy(
                    logits, target_tokens[:, i]
                )
                losses.append(loss)

            return {
                "decoder_loss_1": losses[0],
                "decoder_loss_2": losses[1],
                "decoder_loss_3": losses[2],
                "decoder_loss_4": losses[3],

                "decoder_scores_1": scores[0],
                "decoder_scores_2": scores[1],
                "decoder_scores_3": scores[2],
                "decoder_scores_4": scores[3],

                "decoder_argmax_1": argmaxes[0],
                "decoder_argmax_2": argmaxes[1],
                "decoder_argmax_3": argmaxes[2],
                "decoder_argmax_4": argmaxes[3],
            }
        else:
            pass

    def _apply_encoder(
            self,
            embeddings,  # (batch_size, max_seq_len, embedding_dim)
            mask,  # (batch_size, max_seq_len)
            user_embeddings=None
    ):

        assert embeddings.shape[0] == mask.shape[0]
        assert embeddings.shape[1] == mask.shape[1]

        if user_embeddings is not None:
            # embeddings = torch.cat((user_embeddings.unsqueeze(1), embeddings), dim=1)
            # mask = torch.cat(
            #     (torch.ones((batch_size, 1), dtype=torch.bool, device=DEVICE), mask),
            #     dim=1,
            # )
            # seq_len += 1  # TODOPK ask if this is correct
            pass

        embeddings = self._encoder(
            src=embeddings, src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        return embeddings, mask
