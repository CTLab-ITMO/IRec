from collections import defaultdict
import json
import murmurhash
import numpy as np
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

import torch

from irec.data.transforms import Collate, Transform
from irec.data.dataloader import DataLoader

from data import Dataset



# ПУТИ

IREC_PATH = '../../'
INTERACTIONS_PATH = os.path.join(IREC_PATH, 'data/Beauty/inter.json')
SEMANTIC_MAPPING_PATH = os.path.join(IREC_PATH, 'results/rqvae_beauty_best_clusters_colisionless.json')
TRAIN_BATCHES_DIR = os.path.join(IREC_PATH, 'data/Beauty/tiger_train_batches/')
VALID_BATCHES_DIR = os.path.join(IREC_PATH, 'data/Beauty/tiger_valid_batches/')
EVAL_BATCHES_DIR = os.path.join(IREC_PATH, 'data/Beauty/tiger_eval_batches/')


# ОСТАЛЬНОЕ

SEED_VALUE = 42
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


MAX_SEQ_LEN = 20
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 1024
NUM_USER_HASH = 2000
CODEBOOK_SIZE = 256
NUM_CODEBOOKS = 4

UNIFIED_VOCAB_SIZE = CODEBOOK_SIZE * NUM_CODEBOOKS + NUM_USER_HASH + 10  # 10 for utilities
PAD_TOKEN_ID = UNIFIED_VOCAB_SIZE - 1,
EOS_TOKEN_ID = UNIFIED_VOCAB_SIZE - 2,
DECODER_START_TOKEN_ID = UNIFIED_VOCAB_SIZE - 3,



class TigerProcessing(Transform):
    def __call__(self, batch):
        input_semantic_ids, attention_mask = batch['item.semantic.padded'], batch['item.semantic.mask']
        batch_size = attention_mask.shape[0]

        input_semantic_ids[~attention_mask] = PAD_TOKEN_ID  # TODO ???

        input_semantic_ids = np.concatenate([
            input_semantic_ids,
            NUM_CODEBOOKS * CODEBOOK_SIZE + batch['user.hashed.ids'][:, None]
        ], axis=-1)

        attention_mask = np.concatenate([
            attention_mask,
            np.ones((batch_size, 1), dtype=attention_mask.dtype)
        ], axis=-1)

        batch['input.data'] = input_semantic_ids
        batch['input.mask'] = attention_mask

        target_semantic_ids = batch['labels.semantic.padded']
        target_semantic_ids = np.concatenate([
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


def save_batches_to_arrow(batches, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    for batch_idx, batch in enumerate(batches):
        length_groups = defaultdict(dict)
        metadata_groups = defaultdict(dict)

        for key, value in batch.items():
            length = len(value)

            metadata_groups[length][f'{key}_shape'] = str(value.shape)
            metadata_groups[length][f'{key}_dtype'] = str(value.dtype)

            if value.ndim == 1:
                # 1D массив - сохраняем как есть
                length_groups[length][key] = value
            elif value.ndim == 2:
                # 2D массив - используем list of lists
                length_groups[length][key] = value.tolist()
            else:
                # >2D массив - flatten и сохраняем shape
                length_groups[length][key] = value.flatten()

        for length, fields in length_groups.items():
            arrow_dict = {}
            for k, v in fields.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                    # List of lists (2D)
                    arrow_dict[k] = pa.array(v)
                else:
                    arrow_dict[k] = pa.array(v)

            table = pa.table(arrow_dict)
            if length in metadata_groups:
                table = table.replace_schema_metadata(metadata_groups[length])

            feather.write_feather(
                table,
                output_dir / f"batch_{batch_idx:06d}_len_{length}.arrow",
                compression='lz4'
            )

            # arrow_dict = {k: pa.array(v) for k, v in fields.items()}
            # table = pa.table(arrow_dict)

            # feather.write_feather(
            #     table,
            #     output_dir / f"batch_{batch_idx:06d}_len_{length}.arrow",
            #     compression='lz4'
            # )


def main():
    data = Dataset.create(
        inter_json_path=INTERACTIONS_PATH,
        max_sequence_length=MAX_SEQ_LEN,
        sampler_type='tiger',
        is_extended=True
    )

    with open(SEMANTIC_MAPPING_PATH, 'r') as f:
        mappings = json.load(f)

    train_dataset, valid_dataset, eval_dataset = data.get_datasets()

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    ) \
        .map(Collate()) \
        .map(UserHashing(NUM_USER_HASH)) \
        .map(SemanticIdsMapper(mappings, names=['item', 'labels'])) \
        .map(ToMasked('item.semantic', is_right_aligned=True)) \
        .map(ToMasked('labels.semantic', is_right_aligned=True)) \
        .map(TigerProcessing())

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        drop_last=False
    ) \
        .map(Collate()) \
        .map(UserHashing(NUM_USER_HASH)) \
        .map(SemanticIdsMapper(mappings, names=['item', 'labels'])) \
        .map(ToMasked('item.semantic', is_right_aligned=True)) \
        .map(ToMasked('labels.semantic', is_right_aligned=True)) \
        .map(ToMasked('visited', is_right_aligned=True)) \
        .map(TigerProcessing())

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        drop_last=False
    ) \
        .map(Collate()) \
        .map(UserHashing(NUM_USER_HASH)) \
        .map(SemanticIdsMapper(mappings, names=['item', 'labels'])) \
        .map(ToMasked('item.semantic', is_right_aligned=True)) \
        .map(ToMasked('labels.semantic', is_right_aligned=True)) \
        .map(ToMasked('visited', is_right_aligned=True)) \
        .map(TigerProcessing())

    train_batches = []
    for train_batch in train_dataloader:
        train_batches.append(train_batch)
    save_batches_to_arrow(train_batches, TRAIN_BATCHES_DIR)

    valid_batches = []
    for valid_batch in valid_dataloader:
        valid_batches.append(valid_batch)
    save_batches_to_arrow(valid_batches, VALID_BATCHES_DIR)

    eval_batches = []
    for eval_batch in eval_dataloader:
        eval_batches.append(eval_batch)
    save_batches_to_arrow(eval_batches, EVAL_BATCHES_DIR)



if __name__ == '__main__':
    main()
