import numpy as np
import pickle
import torch

from irec.data.base import BaseDataset
from irec.data.transforms import Transform

from cooc_data import CoocMappingDataset


class AddWeightedCooccurrenceEmbeddings:
    def __init__(self, cooccur_counts, item_id_to_embedding, all_item_ids):
        self.cooccur_counts = cooccur_counts
        self.item_id_to_embedding = item_id_to_embedding
        self.all_item_ids = all_item_ids
        self.call_count = 0

    def __call__(self, batch):
        self.call_count += 1
        item_ids = batch['item_id']
        cooccurrence_embeddings = []
        
        for idx, item_id in enumerate(item_ids):
            item_id_val = int(item_id.item()) if torch.is_tensor(item_id) else int(item_id)
            
            counter = self.cooccur_counts.get(item_id_val)
            if counter and len(counter) > 0:
                cooc_ids, freqs = zip(*counter.items())
                freqs_array = np.array(freqs, dtype=np.float32)
                probs = freqs_array / freqs_array.sum()
                cooc_id = np.random.choice(cooc_ids, p=probs)
                
            else:
                cooc_id = np.random.choice(self.all_item_ids)
                if self.call_count % 10 == 0 and idx < 5:
                    print(f"  idx={idx}: item_id={item_id_val} fallback random")
            
            cooc_emb = self.item_id_to_embedding.get(cooc_id, batch['embedding'][0])
            cooccurrence_embeddings.append(cooc_emb)

        batch['cooccurrence_embedding'] = torch.stack(cooccurrence_embeddings)
        return batch

