import numpy as np
import torch
from typing import Dict, List
import time

import torch
from typing import Dict, List

class AddWeightedCooccurrenceEmbeddingsVectorized:
    def __init__(
        self,
        cooccur_counts: Dict[int, Dict[int, int]],
        item_id_to_embedding: Dict[int, torch.Tensor],
        all_item_ids: List[int],
        device: torch.device,
        max_neighbors: int = 128,
        seed: int = 42,
    ):
        self.device = device
        self.max_neighbors = max_neighbors
        torch.manual_seed(seed)
        
        self.all_item_ids = torch.tensor(sorted(all_item_ids), dtype=torch.long, device=device)
        self.num_items = len(self.all_item_ids)
        
        # 2. Эмбеддинги
        emb_dim = list(item_id_to_embedding.values())[0].shape[0]
        self.embedding_matrix = torch.zeros((self.num_items, emb_dim), dtype=torch.float32, device=device)

        id_to_idx = {iid.item(): i for i, iid in enumerate(self.all_item_ids.cpu())}
        
        for iid, emb in item_id_to_embedding.items():
            if iid in id_to_idx:
                self.embedding_matrix[id_to_idx[iid]] = emb.to(device)


        self.neighbors = torch.randint(0, self.num_items, (self.num_items, max_neighbors), device=device)
        self.probs = torch.full((self.num_items, max_neighbors), 1.0 / max_neighbors, device=device)
        
        neighbors_cpu = self.neighbors.cpu()
        probs_cpu = self.probs.cpu()
        
        for item_id, neighbors_dict in cooccur_counts.items():
            if item_id not in id_to_idx: continue
            idx = id_to_idx[item_id]
            
            if not neighbors_dict: continue
            
            top_k = sorted(neighbors_dict.items(), key=lambda x: x[1], reverse=True)[:max_neighbors]
            ids, counts = zip(*top_k)

            valid_pairs = []
            for n_id, c in zip(ids, counts):
                n_idx = id_to_idx.get(n_id, -1)
                if n_idx != -1:
                    valid_pairs.append((n_idx, c))
            
            if not valid_pairs: continue
            
            final_indices, final_counts = zip(*valid_pairs)
            k_len = len(final_indices)
            
            count_tensor = torch.tensor(final_counts, dtype=torch.float32)
            prob_tensor = count_tensor / count_tensor.sum()
            
            neighbors_cpu[idx, :k_len] = torch.tensor(final_indices, dtype=torch.long)
            probs_cpu[idx, :k_len] = prob_tensor
            
            if k_len < max_neighbors:
                probs_cpu[idx, k_len:] = 0.0
            
        self.neighbors = neighbors_cpu.to(device)
        self.probs = probs_cpu.to(device)
        
    def __call__(self, batch):
        item_ids = batch['item_id'].to(self.device)
        
        indices = torch.searchsorted(self.all_item_ids, item_ids)
        indices = indices.clamp(max=self.num_items - 1)
        
        found_mask = (self.all_item_ids[indices] == item_ids)
        
        if not found_mask.all():
            missing_count = (~found_mask).sum().item()
            missing_examples = item_ids[~found_mask][:5].tolist()
            print(f"[WARNING] Batch contains {missing_count} unknown items! Examples: {missing_examples}")
            print(f"          Assigning RANDOM embeddings for unknown items.")

        # кого не нашли, подменим индекс на 0 временно
        safe_indices = indices.clone()
        safe_indices[~found_mask] = 0 
        
        batch_probs = self.probs[safe_indices] # (B, max_neighbors)
        neighbor_local_indices = torch.multinomial(batch_probs, num_samples=1).squeeze(1) # (B)
        selected_neighbor_indices = self.neighbors[safe_indices, neighbor_local_indices] # (B)
        cooc_embeddings = self.embedding_matrix[selected_neighbor_indices]
        
        if not found_mask.all():
            # случайные индексы айтемов без истории
            random_indices = torch.randint(0, self.num_items, (item_ids.shape[0],), device=self.device)
            random_embeddings = self.embedding_matrix[random_indices]
            
            # шум
            cooc_embeddings = torch.where(
                found_mask.unsqueeze(1), 
                cooc_embeddings, 
                random_embeddings
            )
        
        batch['cooccurrence_embedding'] = cooc_embeddings
        return batch
