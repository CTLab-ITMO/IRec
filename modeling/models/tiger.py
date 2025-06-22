import torch
import torch.nn as nn

from models import TorchModel
from utils import get_activation_function, create_masked_tensor, DEVICE


class TigerModel(TorchModel, config_name="tiger"):
    def __init__(
            self,
            sequence_prefix,
            embedding_dim,
            codebook_size,
            num_positions,
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
        self._embedding_dim = embedding_dim
        self._codebook_size = codebook_size
        self._num_positions = num_positions
        self._num_heads = num_heads
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dim_feedforward = dim_feedforward
        self._dropout = nn.Dropout(dropout)
        self._layer_norm_eps = layer_norm_eps

        self._sem_id_len = 4

        self.position_embeddings = nn.Embedding(num_embeddings=self._num_positions, embedding_dim=self._embedding_dim,
                                                device=DEVICE)

        self.sem_id_position_embeddings = nn.Embedding(num_embeddings=self._sem_id_len,
                                                       embedding_dim=self._embedding_dim, device=DEVICE)

        self.bos_embedding = nn.Parameter(torch.randn(self._embedding_dim, device=DEVICE))

        self.codebook_embeddings = nn.Embedding(num_embeddings=(self._codebook_size * self._sem_id_len),
                                                embedding_dim=self._embedding_dim, device=DEVICE)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._embedding_dim,
            nhead=self._num_heads,
            dim_feedforward=self._dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=self._layer_norm_eps,
            batch_first=True,
            device=DEVICE
        )

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self._embedding_dim,
            nhead=self._num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=self._layer_norm_eps,
            batch_first=True,
            device=DEVICE
        )

        self._decoder = nn.TransformerDecoder(
            transformer_decoder_layer, num_decoder_layers
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_encoder_layers)

        # self._layernorm = nn.LayerNorm(self._embedding_dim, eps=layer_norm_eps)

        self._init_weights(initializer_range)

    def _embed_semantic_tokens(self, sem_ids: torch.LongTensor) -> torch.Tensor:
        """
        sem_ids: (N,)
        embeds: (N, embedding_dim)
        """

        positions = torch.arange(sem_ids.size(0), device=DEVICE) % self._sem_id_len  # (N,)
        offsets = positions * self._codebook_size
        assert offsets.shape == sem_ids.shape
        return self.codebook_embeddings(offsets + sem_ids)  # (N , embedding_dim)

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

    def _get_last_sem_ids_mask(self, all_sample_lengths: torch.Tensor) -> torch.Tensor:
        """Создает маску для последних sem_id_len токенов каждой последовательности"""
        total_tokens = all_sample_lengths.sum().item()
        tgt_end_idx = torch.cumsum(all_sample_lengths, dim=0)
        tgt_start_idx = tgt_end_idx - self._sem_id_len

        mask_flat_extended = torch.zeros(total_tokens + 1, dtype=torch.int, device=DEVICE)
        mask_flat_extended[tgt_start_idx] += 1
        mask_flat_extended[tgt_end_idx] -= 1

        mask_flat = torch.cumsum(mask_flat_extended, dim=0)[:total_tokens]
        return mask_flat.bool()

    def _prepare_sem_id_batch(
            self,
            embeddings_flat: torch.Tensor,  # (total_tokens, embedding_dim)
            lengths: torch.LongTensor  # (batch_size,)
    ):
        batch_size = lengths.size(0)
        sem_id_len = self._sem_id_len

        # маска последних sem_id_len каждого батча
        decoder_mask_flat = self._get_last_sem_ids_mask(lengths)

        # разделение плоского тензора
        encoder_emb_flat = embeddings_flat[~decoder_mask_flat]
        decoder_emb_flat = embeddings_flat[decoder_mask_flat]

        # эмбеддинги уже с добавленной размерностью max_seq_len и позиционными
        encoder_embeddings, encoder_mask = self._create_encoder_tensors(encoder_emb_flat, lengths, sem_id_len)
        decoder_embeddings = self._create_decoder_tensors(decoder_emb_flat, batch_size, sem_id_len)

        # BOS
        encoder_embeddings, encoder_mask = self._add_bos_to_encoder(encoder_embeddings, encoder_mask, batch_size)
        decoder_embeddings = self._add_bos_to_decoder(decoder_embeddings, batch_size)

        return encoder_embeddings, encoder_mask, decoder_embeddings

    def _create_encoder_tensors(self, encoder_emb_flat, lengths, sem_id_len):
        encoder_lengths = lengths - sem_id_len
        encoder_embeddings, encoder_mask = create_masked_tensor(encoder_emb_flat, encoder_lengths)

        # позиционные эмбеддинги
        pos_emb = self._get_position_embeddings(encoder_mask)
        sem_pos_emb = self._get_sem_ids_position_embeddings(encoder_mask)
        encoder_embeddings += pos_emb + sem_pos_emb

        return encoder_embeddings, encoder_mask

    def _create_decoder_tensors(self, decoder_emb_flat, batch_size, sem_id_len):
        decoder_embeddings = decoder_emb_flat.view(batch_size, sem_id_len, -1)

        # позиционные эмбеддинги (только семантические)
        sem_pos_ids = torch.arange(sem_id_len, device=DEVICE).expand(batch_size, -1)
        sem_pos_emb = self.sem_id_position_embeddings(sem_pos_ids)
        decoder_embeddings += sem_pos_emb

        return decoder_embeddings

    def _add_bos_to_encoder(self, encoder_embeddings, encoder_mask, batch_size):
        bos = self.bos_embedding.view(1, 1, -1).expand(batch_size, 1, -1)
        new_encoder_embeddings = torch.cat([bos, encoder_embeddings], dim=1)
        new_mask = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=DEVICE),
            encoder_mask
        ], dim=1)
        return new_encoder_embeddings, new_mask

    def _add_bos_to_decoder(self, decoder_embeddings, batch_size):
        bos = self.bos_embedding.view(1, 1, -1).expand(batch_size, 1, -1)
        return torch.cat([bos, decoder_embeddings], dim=1)

    def forward(self, inputs):
        all_sample_events = inputs[
            "semantic_{}.ids".format(self._sequence_prefix)
        ]  # (all_batch_events)
        all_sample_lengths = inputs[
            "semantic_{}.length".format(self._sequence_prefix)
        ] # (batch_size)
        assert all_sample_events.shape[0] == sum(all_sample_lengths)
        embeddings_flat = self._embed_semantic_tokens(all_sample_events)

        assert embeddings_flat.shape[0] == sum(all_sample_lengths)

        (encoder_input_emb,  # (batch_size, seq_len - sem_id_len + 1, embedding_dim)
         encoder_input_mask,  # (batch_size, seq_len - sem_id_len + 1)
         decoder_input_embs) = (  # (batch_size, sem_id_len + 1, embedding_dim)
            self._prepare_sem_id_batch(embeddings_flat, all_sample_lengths)
        )

        after_encoder_emb, after_encoder_mask = self._apply_encoder(encoder_input_emb, encoder_input_mask)

        if self.training:
            # последние sem ids
            target_tokens_mask = self._get_last_sem_ids_mask(all_sample_lengths)
            target_tokens = all_sample_events[target_tokens_mask].view(-1, self._sem_id_len)  # (batch_size, sem_id_len)

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
                weights = self.codebook_embeddings.weight[i * self._codebook_size: (i + 1) * self._codebook_size]
                logits = torch.matmul(
                    decoder_output[:, i, :], weights.t()
                )  # (batch_size, codebook_size)
                scores.append(logits)

                pred_tokens = torch.argmax(logits, dim=-1)  # (batch_size,)
                argmaxes.append(pred_tokens)

                loss = nn.functional.cross_entropy(logits, target_tokens[:, i])
                losses.append(loss)

            return {
                "decoder_loss_1": losses[0],  # (1, )
                "decoder_loss_2": losses[1],  # (1, )
                "decoder_loss_3": losses[2],  # (1, )
                "decoder_loss_4": losses[3],  # (1, )

                "decoder_scores_1": scores[0],  # (batch_size, codebook_size)
                "decoder_scores_2": scores[1],  # (batch_size, codebook_size)
                "decoder_scores_3": scores[2],  # (batch_size, codebook_size)
                "decoder_scores_4": scores[3],  # (batch_size, codebook_size)

                "decoder_argmax_1": argmaxes[0],  # (batch_size, )
                "decoder_argmax_2": argmaxes[1],  # (batch_size, )
                "decoder_argmax_3": argmaxes[2],  # (batch_size, )
                "decoder_argmax_4": argmaxes[3],  # (batch_size, )
            }
        else:
            batch_size = encoder_input_emb.size(0)

            tgt = self.bos_embedding.view(1, 1, -1).expand(batch_size, 1, -1)  # (batch_size, 1, embedding_dim)

            memory_key_padding_mask = ~after_encoder_mask

            argmaxes = []
            scores = []

            for step in range(self._sem_id_len):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt.size(1), device=DEVICE
                )  # (L, L)

                decoder_output = self._decoder(
                    tgt=tgt,
                    memory=after_encoder_emb,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )  # (batch_size, L, embedding_dim)

                last_output = decoder_output[:, -1:, :]  # (batch_size, 1, embedding_dim)

                weights = self.codebook_embeddings.weight[step * self._codebook_size: (step + 1) * self._codebook_size]
                logits = torch.matmul(
                    last_output,
                    weights.t()
                ).squeeze(1)  # (batch_size, codebook_size)

                scores.append(logits)
                pred_token = torch.argmax(logits, dim=-1)  # (batch_size,)
                argmaxes.append(pred_token)

                if step < self._sem_id_len - 1:
                    next_embed = self.codebook_embeddings(
                        step * self._codebook_size + pred_token)  # (batch_size, embedding_dim)

                    pos_emb = self.sem_id_position_embeddings(
                        torch.tensor([step], device=DEVICE)
                    ).expand(batch_size, -1)
                    next_embed += pos_emb

                    next_embed = next_embed.unsqueeze(1)  # (batch_size, 1, embedding_dim)
                    tgt = torch.cat([tgt, next_embed], dim=1)

            return {
                "decoder_scores_1": scores[0],
                "decoder_scores_2": scores[1],
                "decoder_scores_3": scores[2],
                "decoder_scores_4": scores[3],

                "decoder_argmax_1": argmaxes[0],
                "decoder_argmax_2": argmaxes[1],
                "decoder_argmax_3": argmaxes[2],
                "decoder_argmax_4": argmaxes[3],
            }

    def _apply_encoder(
            self,
            embeddings,  # (batch_size, max_seq_len, embedding_dim)
            mask,  # (batch_size, max_seq_len)
    ):

        assert embeddings.shape[0] == mask.shape[0]
        assert embeddings.shape[1] == mask.shape[1]

        embeddings = self._encoder(
            src=embeddings, src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        return embeddings, mask

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config["sequence_prefix"],
            embedding_dim=config["embedding_dim"],
            codebook_size=config["codebook_size"],
            num_positions=config["num_positions"],
            num_heads=config.get("num_heads", int(config["embedding_dim"] // 64)),
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            dim_feedforward=config.get("dim_feedforward", 4 * config["embedding_dim"]),
            dropout=config.get("dropout", 0.0),
            initializer_range=config.get("initializer_range", 0.02),
        )
