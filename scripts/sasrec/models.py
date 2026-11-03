import torch
import torch.nn as nn

from irec.models import TorchModel, create_masked_tensor



class SasRecModel(TorchModel):
    def __init__(
            self,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            activation,
            topk_k,
            dropout=0.0,
            layer_norm_eps=1e-9,
            initializer_range=0.02
    ):
        super().__init__()
        self._num_items = num_items
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim
        )

        self._topk_k = topk_k

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

        self._init_weights(initializer_range)

    def forward(self, inputs):
        all_sample_events = inputs['item.ids']  # (total_batch_items)
        all_sample_lengths = inputs['item.length']  # (batch_size)
        embeddings, mask = self._apply_sequential_encoder(all_sample_events, all_sample_lengths)
        all_sample_embeddings = embeddings[mask]

        all_positive_sample_events = inputs['labels.ids']  # (total_batch_items)
        
        if not self.training:
            offsets = torch.cumsum(all_sample_lengths, dim=-1)
            all_sample_embeddings = all_sample_embeddings[offsets - 1]

        all_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)

        # a -- total_batch_items, n -- num_items, d -- embedding_dim
        all_scores = torch.einsum(
            'ad,nd->an',
            all_sample_embeddings,
            all_embeddings
        )  # (total_batch_items, num_items)

        positive_scores = torch.gather(
            input=all_scores,
            dim=1,
            index=all_positive_sample_events[..., None]
        )[:, 0]  # (total_batch_items)

        # Compute loss
        negative_scores = torch.gather(
            input=all_scores,
            dim=1,
            index=torch.randint(
                low=0,
                high=all_scores.shape[1],
                size=all_positive_sample_events.shape,
                device=all_positive_sample_events.device
            )[..., None]
        )[:, 0]  # (total_batch_items)

        with torch.autocast(device_type='cuda', enabled=False):
            loss = self._compute_loss(
                positive_scores.float(),
                negative_scores.float()
            )

        metrics = {
            'loss': loss.detach()
        }

        if not self.training:
            batch_size = all_sample_lengths.shape[0]
            num_items = all_embeddings.shape[0]
            
            padded_items, _ = create_masked_tensor(
                data=all_sample_events,
                lengths=all_sample_lengths,
            )  # (batch_size, max_seq_len)

            visited_mask = torch.zeros(
                batch_size, num_items,
                dtype=torch.bool,
                device=all_sample_events.device
            )

            batch_indices = torch.arange(batch_size, device=all_sample_events.device)[:, None]
            batch_indices = batch_indices.expand(-1, padded_items.shape[1])

            visited_mask.scatter_(
                dim=1,
                index=padded_items.long(),
                value=True
            )

            all_scores = all_scores.masked_fill(visited_mask, float('-inf'))

        positive_position = (all_scores > positive_scores[:, None]).float().sum(dim=-1)  # (batch_size или total_batch_items)
        dcg_score = 1. / (torch.log2(positive_position + 1) + 1.)

        for k in [5, 10, 20]:
            metrics[f'recall@{k}'] = (positive_position < k).float()
            metrics[f'ndcg@{k}'] = torch.where(
                positive_position < k,
                dcg_score, 
                torch.zeros_like(dcg_score)
            ).float()

        return loss, metrics

    def _apply_sequential_encoder(self, events, lengths):
        embeddings = self._item_embeddings(events)  # (all_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings,
            lengths=lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positions = torch.arange(
            start=0, end=seq_len, device=mask.device
        )[None].expand(batch_size, -1)  # (batch_size, seq_len)

        position_embeddings = self._position_embeddings(positions)   # (batch_size, seq_len, embedding_dim)
        position_embeddings[~mask] = 0

        embeddings = embeddings + position_embeddings  # (batch_size, seq_len, embedding_dim)
        embeddings[~mask] = 0
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).bool().to(embeddings.device)  # (seq_len, seq_len)
        embeddings = self._encoder(
            src=embeddings,
            mask=causal_mask,
            src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        return embeddings, mask

    def _compute_loss(self, positive_scores, negative_scores):
        assert positive_scores.shape[0] == negative_scores.shape[0]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            positive_scores, torch.ones_like(positive_scores)
        ) + torch.nn.functional.binary_cross_entropy_with_logits(
            negative_scores, torch.zeros_like(negative_scores)
        )

        return loss
