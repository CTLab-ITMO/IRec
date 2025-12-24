import torch
from transformers import T5ForConditionalGeneration, T5Config, LogitsProcessor

from irec.models import TorchModel


class CorrectItemsLogitsProcessor(LogitsProcessor):
    def __init__(self, num_codebooks, codebook_size, mapping, num_beams, visited_items):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.num_beams = num_beams

        semantic_ids = []
        for i in range(len(mapping)):
            assert len(mapping[str(i)]) == num_codebooks, 'All semantic ids must have the same length'
            semantic_ids.append(mapping[str(i)])
        
        self.index_semantic_ids = torch.tensor(semantic_ids, dtype=torch.long, device=visited_items.device)  # (num_items, semantic_ids)

        batch_size, _ = visited_items.shape

        self.index_semantic_ids = torch.tile(self.index_semantic_ids[None], dims=[batch_size, 1, 1])  # (batch_size, num_items, semantic_ids)

        index = visited_items[..., None].tile(dims=[1, 1, num_codebooks])  # (batch_size, num_rated, semantic_ids)
        self.index_semantic_ids = torch.scatter(
            input=self.index_semantic_ids,
            dim=1,
            index=index,
            src=torch.zeros_like(index)
        )  # (batch_size, num_items, semantic_ids)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        next_sid_codebook_num = (torch.minimum((input_ids[:, -1].max() // self.codebook_size), torch.as_tensor(self.num_codebooks - 1)).item() + 1) % self.num_codebooks
        a = torch.tile(self.index_semantic_ids[:, None, :, next_sid_codebook_num], dims=[1, self.num_beams, 1])  # (batch_size, num_beams, num_items)
        a = a.reshape(a.shape[0] * a.shape[1], a.shape[2])  # (batch_size * num_beams, num_items)

        if next_sid_codebook_num != 0:
            b = torch.tile(self.index_semantic_ids[:, None :, :next_sid_codebook_num], dims=[1, self.num_beams, 1, 1])  # (batch_size, num_beams, num_items, sid_len)
            b = b.reshape(b.shape[0] * b.shape[1], b.shape[2], b.shape[3])  # (batch_size * num_beams, num_items, sid_len)

            current_prefixes = input_ids[:, -next_sid_codebook_num:]  # (batch_size * num_beams, sid_len)
            possible_next_items_mask = (
                torch.eq(current_prefixes[:, None, :], b).long().sum(dim=-1) == next_sid_codebook_num
            )  # (batch_size * num_beams, num_items)
            a[~possible_next_items_mask] = (next_sid_codebook_num + 1) * self.codebook_size

        scores_mask = torch.zeros_like(scores).bool()  # (batch_size * num_beams, num_items)
        scores_mask = torch.scatter_add(
            input=scores_mask,
            dim=-1,
            index=a,
            src=torch.ones_like(a).bool()
        )
        
        scores[:, :next_sid_codebook_num * self.codebook_size] = -torch.inf
        scores[:, (next_sid_codebook_num + 1) * self.codebook_size:] = -torch.inf
        scores[~(scores_mask.bool())] = -torch.inf
        
        return scores


class TigerModel(TorchModel):
    def __init__(
            self,
            embedding_dim,
            codebook_size,
            sem_id_len,
            num_positions,
            user_ids_count,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            num_beams=100,
            num_return_sequences=20,
            d_kv=64,
            layer_norm_eps=1e-6,
            activation='relu',
            dropout=0.1,
            initializer_range=0.02,
            logits_processor=None,
            use_microbatching=False,
            microbatch_size=128
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._codebook_size = codebook_size
        self._num_positions = num_positions
        self._num_heads = num_heads
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dim_feedforward = dim_feedforward
        self._num_beams = num_beams
        self._num_return_sequences = num_return_sequences
        self._d_kv = d_kv
        self._layer_norm_eps = layer_norm_eps
        self._activation = activation
        self._dropout = dropout
        self._sem_id_len = sem_id_len
        self.user_ids_count = user_ids_count
        self.logits_processor = logits_processor
        self._use_microbatching = use_microbatching
        self._microbatch_size = microbatch_size

        unified_vocab_size = codebook_size * self._sem_id_len + self.user_ids_count + 10  # 10 for utilities
        self.config = T5Config(
            vocab_size=unified_vocab_size,
            d_model=self._embedding_dim,
            d_kv=self._d_kv,
            d_ff=self._dim_feedforward,
            num_layers=self._num_encoder_layers,
            num_decoder_layers=self._num_decoder_layers,
            num_heads=self._num_heads,
            dropout_rate=self._dropout,
            is_encoder_decoder=True,
            use_cache=False,
            pad_token_id=unified_vocab_size - 1,
            eos_token_id=unified_vocab_size - 2,
            decoder_start_token_id=unified_vocab_size - 3,
            layer_norm_epsilon=self._layer_norm_eps,
            feed_forward_proj=self._activation,
            tie_word_embeddings=False
        )
        self.model = T5ForConditionalGeneration(config=self.config)
        self._init_weights(initializer_range)

        self.model = torch.compile(
            self.model, 
            mode='reduce-overhead',
            fullgraph=False,
            dynamic=True
        )

    def forward(self, inputs):
        input_semantic_ids = inputs['input.data']
        attention_mask = inputs['input.mask']
        target_semantic_ids = inputs['output.data']

        decoder_input_ids = target_semantic_ids[:, :-1].contiguous()
        labels = target_semantic_ids[:, 1:].contiguous()

        model_output = self.model(
            input_ids=input_semantic_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        loss = model_output['loss']

        metrics = {'loss': loss.detach()}
        
        if not self.training and not self._use_microbatching:
            visited_batch = inputs['visited.padded']

            output = self.model.generate(
                input_ids=input_semantic_ids,
                attention_mask=attention_mask,
                num_beams=self._num_beams,
                num_return_sequences=self._num_return_sequences,
                max_length=self._sem_id_len + 1,
                decoder_start_token_id=self.config.decoder_start_token_id,
                eos_token_id=self.config.eos_token_id,
                pad_token_id=self.config.pad_token_id,
                do_sample=False,
                early_stopping=False,
                logits_processor=[self.logits_processor(visited_items=visited_batch)] if self.logits_processor is not None else [],
            )
            
            predictions = output[:, 1:].reshape(-1, self._num_return_sequences, self._sem_id_len)

            all_hits = (torch.eq(predictions, labels[:, None]).sum(dim=-1))  # (batch_size, top_k)
        elif not self.training and self._use_microbatching:
            visited_batch = inputs['visited.padded']
            batch_size = input_semantic_ids.shape[0]

            inference_batch_size = self._microbatch_size  # вместо полного batch_size

            all_predictions = []
            all_labels = []
            # print(f"start to infer batch of shape {input_semantic_ids.shape} with new batch {inference_batch_size}")
            for batch_idx in range(0, batch_size, inference_batch_size):
                batch_end = min(batch_idx + inference_batch_size, batch_size)
                batch_slice = slice(batch_idx, batch_end)

                input_ids_batch = input_semantic_ids[batch_slice]
                attention_mask_batch = attention_mask[batch_slice]
                visited_batch_subset = visited_batch[batch_slice]
                labels_batch = labels[batch_slice]

                with torch.inference_mode():
                    output = self.model.generate(
                        input_ids=input_ids_batch,
                        attention_mask=attention_mask_batch,
                        num_beams=self._num_beams,
                        num_return_sequences=self._num_return_sequences,
                        max_length=self._sem_id_len + 1,
                        decoder_start_token_id=self.config.decoder_start_token_id,
                        eos_token_id=self.config.eos_token_id,
                        pad_token_id=self.config.pad_token_id,
                        do_sample=False,
                        early_stopping=False,
                        logits_processor=[self.logits_processor(visited_items=visited_batch_subset)] if self.logits_processor is not None else [],
                    )

                predictions_batch = output[:, 1:].reshape(-1, self._num_return_sequences, self._sem_id_len)
                all_predictions.append(predictions_batch)
                all_labels.append(labels_batch)
            # print("end infer of batch")

            predictions = torch.cat(all_predictions, dim=0)  # (batch_size, num_return_sequences, sem_id_len)
            labels_full = torch.cat(all_labels, dim=0)  # (batch_size, sem_id_len)
            all_hits = (torch.eq(predictions, labels_full[:, None]).sum(dim=-1))  # (batch_size, top_k)

        if not self.training:
            for k in [5, 10, 20]:
                hits = (all_hits[:, :k] == self._sem_id_len).float() # (batch_size, k)
                recall = hits.sum(dim=-1)  # (batch_size)
                discount_factor = 1 / torch.log2(torch.arange(1, k + 1, 1).float() + 1.).to(hits.device)  # (k)

                metrics[f'recall@{k}'] = recall.cpu().float()
                metrics[f'ndcg@{k}'] = torch.einsum('bk,k->b', hits, discount_factor).cpu().float()

        return loss, metrics