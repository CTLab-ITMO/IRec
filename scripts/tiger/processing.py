import murmurhash
import numpy as np
import torch
from irec.data.transforms import Transform

NUM_USER_HASH = 2000
CODEBOOK_SIZE = 256
NUM_CODEBOOKS = 4
UNIFIED_VOCAB_SIZE = CODEBOOK_SIZE * NUM_CODEBOOKS + NUM_USER_HASH + 10
PAD_TOKEN_ID = UNIFIED_VOCAB_SIZE - 1
DECODER_START_TOKEN_ID = UNIFIED_VOCAB_SIZE - 3

class TigerProcessing(Transform):
    def __call__(self, batch):
        input_semantic_ids, attention_mask = batch['item.semantic.padded'], batch['item.semantic.mask']
        batch_size = attention_mask.shape[0]

        input_semantic_ids[~attention_mask] = PAD_TOKEN_ID

        input_semantic_ids = np.concat([
            input_semantic_ids,
            NUM_CODEBOOKS * CODEBOOK_SIZE + batch['user.hashed.ids'][:, None]
        ], axis=-1)

        attention_mask = np.concat([
            attention_mask,
            np.ones((batch_size, 1), dtype=attention_mask.dtype)
        ], axis=-1)

        batch['input.data'] = input_semantic_ids
        batch['input.mask'] = attention_mask

        target_semantic_ids = batch['labels.semantic.padded']
        target_semantic_ids = np.concat([
            np.ones(
                (batch_size, 1),
                dtype=np.int64,
            ) * DECODER_START_TOKEN_ID,
            target_semantic_ids
        ], axis=-1)

        batch['output.data'] = target_semantic_ids

        return batch


class ToMasked(Transform):
    def __init__(self, prefix, is_right_aligned=False):
        self._prefix = prefix
        self._is_right_aligned = is_right_aligned
    
    def __call__(self, batch):
        data = batch[f'{self._prefix}.ids']
        lengths = batch[f'{self._prefix}.length']

        batch_size = lengths.shape[0]
        max_sequence_length = int(lengths.max())

        if len(data.shape) == 1:  # only indices
            padded_tensor = np.zeros(
                (batch_size, max_sequence_length),
                dtype=data.dtype
            )  # (batch_size, max_seq_len)
        else:
            assert len(data.shape) == 2  # embeddings
            padded_tensor = np.zeros(
                (batch_size, max_sequence_length, data.shape[-1]),
                dtype=data.dtype
            )  # (batch_size, max_seq_len, emb_dim)
        
        mask = np.arange(max_sequence_length)[None] < lengths[:, None]

        if self._is_right_aligned:
            mask = np.flip(mask, axis=-1)

        padded_tensor[mask] = data

        batch[f'{self._prefix}.padded'] = padded_tensor
        batch[f'{self._prefix}.mask'] = mask

        return batch


class SemanticIdsMapper(Transform):
    def __init__(self, mapping, names=[]):
        super().__init__()
        self._mapping = mapping
        self._names = names

        data = []
        for i in range(len(mapping)):
            data.append(mapping[str(i)])
        self._mapping_tensor = torch.tensor(data, dtype=torch.long)
        self._semantic_length = self._mapping_tensor.shape[-1]
    
    def __call__(self, batch):
        for name in self._names:
            if f'{name}.ids' in batch:  
                ids = batch[f'{name}.ids']
                lengths = batch[f'{name}.length']
                assert ids.min() >= 0
                assert ids.max() < self._mapping_tensor.shape[0]
                batch[f'{name}.semantic.ids'] = self._mapping_tensor[ids].flatten().numpy()
                batch[f'{name}.semantic.length'] = lengths * self._semantic_length

        return batch


class UserHashing(Transform):
    def __init__(self, hash_size):
        super().__init__()
        self._hash_size = hash_size
    
    def __call__(self, batch):
        batch['user.hashed.ids'] = np.array([murmurhash.hash(str(x)) % self._hash_size for x in batch['user.ids']], dtype=np.int64)
        return batch


