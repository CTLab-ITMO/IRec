import numpy as np
import pickle

from irec.data.base import BaseDataset
from irec.data.transforms import Transform


import polars as pl

class InteractionsDatasetParquet(BaseDataset):
    def __init__(self, data_path, max_items=None):
        self.df = pl.read_parquet(data_path)
        assert 'uid' in self.df.columns, "Missing 'uid' column"
        assert 'item_ids' in self.df.columns, "Missing 'item_ids' column"
        print(f"Dataset loaded: {len(self.df)} users")

        if max_items is not None:
            self.df = self.df.with_columns(
                pl.col("item_ids").list.slice(-max_items).alias("item_ids")
            )

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        return {
            'user_id': row['uid'],
            'item_ids': np.array(row['item_ids'], dtype=np.uint32),
        }

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class EmbeddingDatasetParquet(BaseDataset):
    def __init__(self, data_path):
        self.df = pl.read_parquet(data_path)
        self.item_ids = np.array(self.df['item_id'], dtype=np.int64)
        self.embeddings = np.array(self.df['embedding'].to_list(), dtype=np.float32)
        print(f"embedding dim: {self.embeddings[0].shape}")

    def __getitem__(self, idx):
        index = self.item_ids[idx]
        tensor_emb = self.embeddings[idx]
        return {
            'item_id': index,
            'embedding': tensor_emb,
            'embedding_dim': len(tensor_emb)
        }

    def __len__(self):
        return len(self.embeddings)


class EmbeddingDataset(BaseDataset):
    def __init__(self, data_path):
        self.data_path = data_path
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.item_ids = np.array(self.data['item_id'], dtype=np.int64)
        self.embeddings = np.array(self.data['embedding'], dtype=np.float32)

    def __getitem__(self, idx):
        index = self.item_ids[idx]
        tensor_emb = self.embeddings[idx]
        return {
            'item_id': index,
            'embedding': tensor_emb,
            'embedding_dim': len(tensor_emb)
        }

    def __len__(self):
        return len(self.embeddings)


class ProcessEmbeddings(Transform):
    def __init__(self, embedding_dim, keys):
        self.embedding_dim = embedding_dim
        self.keys = keys
    
    def __call__(self, batch):
        for key in self.keys:
            batch[key] = batch[key].reshape(-1, self.embedding_dim)
        return batch