from collections import defaultdict
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

import torch

from irec.data.transforms import Collate
from irec.data.dataloader import DataLoader

from data import Dataset


NUM_EPOCHS = 300
MAX_SEQ_LEN = 20
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SiZE = 256

IREC_PATH = '../../'


def save_batches_to_arrow(batches, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    
    for batch_idx, batch in enumerate(batches):
        length_groups = defaultdict(dict)
        
        for key, value in batch.items():
            length = len(value)
            length_groups[length][key] = value
        
        for length, fields in length_groups.items():
            arrow_dict = {k: pa.array(v) for k, v in fields.items()}
            table = pa.table(arrow_dict)
            
            feather.write_feather(
                table,
                output_dir / f"batch_{batch_idx:06d}_len_{length}.arrow",
                compression='lz4'
            )

def main():

    data = Dataset.create(
        inter_json_path=os.path.join(IREC_PATH, 'data/Beauty/inter.json'),
        max_sequence_length=MAX_SEQ_LEN,
        sampler_type='sasrec',
        is_extended=False
    )

    train_dataset, valid_dataset, eval_dataset = data.get_datasets()

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    ).map(Collate()).repeat(NUM_EPOCHS)

    valid_dataloder = DataLoader(
        dataset=valid_dataset,
        batch_size=VALID_BATCH_SiZE,
        shuffle=False,
        drop_last=False
    ).map(Collate())

    eval_dataloder = DataLoader(
        dataset=eval_dataset,
        batch_size=VALID_BATCH_SiZE,
        shuffle=False,
        drop_last=False
    ).map(Collate())

    train_batches = []
    for train_batch in train_dataloader:
        train_batches.append(train_batch)
    save_batches_to_arrow(train_batches, os.path.join(IREC_PATH, 'data/Beauty/sasrec_train/'))
    
    valid_batches = []
    for valid_batch in valid_dataloder:
        valid_batches.append(valid_batch)
    save_batches_to_arrow(valid_batches, os.path.join(IREC_PATH, 'data/Beauty/sasrec_valid/'))
    
    eval_batches = []
    for eval_batch in eval_dataloder:
        eval_batches.append(eval_batch)
    save_batches_to_arrow(eval_batches, os.path.join(IREC_PATH, 'data/Beauty/sasrec_eval/'))


if __name__ == '__main__':
    main()
