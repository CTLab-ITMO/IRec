from collections import defaultdict
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

from irec.data.transforms import Collate
from irec.data.dataloader import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import Dataset

# ПУТИ
IREC_PATH = '../../../'

INTERACTIONS_TRAIN_PATH = "/home/jovyan/IRec/sigir/lsvd_data_filtered/15-ts-ows/base_with_gap_interactions_grouped.parquet"
INTERACTIONS_VALID_PATH = "/home/jovyan/IRec/sigir/lsvd_data_filtered/15-ts-ows/val_interactions_grouped.parquet"
INTERACTIONS_TEST_PATH = "/home/jovyan/IRec/sigir/lsvd_data_filtered/15-ts-ows/test_interactions_grouped.parquet"

TRAIN_BATCHES_DIR = os.path.join(IREC_PATH, 'data/lsvd-2/sasrec_base_gap/train_batches/')
VALID_BATCHES_DIR = os.path.join(IREC_PATH, 'data/lsvd-2/sasrec_base_gap/valid_batches/')
EVAL_BATCHES_DIR = os.path.join(IREC_PATH, 'data/lsvd-2/sasrec_base_gap/eval_batches/')

NUM_EPOCHS = 300
MAX_SEQ_LEN = 20
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 1024

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
    data = Dataset.create_timestamp_based_parquet(
        train_parquet_path=INTERACTIONS_TRAIN_PATH,
        validation_parquet_path=INTERACTIONS_VALID_PATH,
        test_parquet_path=INTERACTIONS_TEST_PATH,
        max_sequence_length=MAX_SEQ_LEN,
        sampler_type='sasrec',
        min_sample_len=2,
        is_extended=False,
        max_train_events=MAX_SEQ_LEN
    )

    train_dataset, valid_dataset, eval_dataset = data.get_datasets()

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    ).map(Collate()).repeat(NUM_EPOCHS)

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        drop_last=False
    ).map(Collate())

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        drop_last=False
    ).map(Collate())

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
