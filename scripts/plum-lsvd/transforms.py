import numpy as np
import pickle
import torch
from typing import Dict, List
import time
from collections import defaultdict, Counter

class AddWeightedCooccurrenceEmbeddings:
    def __init__(self, cooccur_counts, item_id_to_embedding, all_item_ids, top_k):
        self.cooccur_counts = cooccur_counts
        self.item_id_to_embedding = item_id_to_embedding
        self.all_item_ids = all_item_ids
        self.call_count = 0
        self.top_k = top_k
        
        # Предвычисляем top_k для каждого item_id
        self._top_k_cache = {}
        self._build_top_k_cache()
    
    def _build_top_k_cache(self):
        """Предвычисляет top-k соседей для каждого item_id"""
        for item_id, counter in self.cooccur_counts.items():
            if counter and len(counter) > 0:
                # Сортируем по частоте и берем top_k
                top_items = counter.most_common(self.top_k)
                cooc_ids, freqs = zip(*top_items)
                freqs_array = np.array(freqs, dtype=np.float32)
                probs = freqs_array / freqs_array.sum()
                
                self._top_k_cache[item_id] = {
                    'cooc_ids': cooc_ids,
                    'probs': probs
                }
    
    def __call__(self, batch):
        self.call_count += 1
        item_ids = batch['item_id']
        cooccurrence_embeddings = []
        
        for idx, item_id in enumerate(item_ids):
            item_id_val = int(item_id.item()) if torch.is_tensor(item_id) else int(item_id)
            
            # Используем предвычисленный top-k кэш
            if item_id_val in self._top_k_cache:
                cache_entry = self._top_k_cache[item_id_val]
                cooc_id = np.random.choice(
                    cache_entry['cooc_ids'], 
                    p=cache_entry['probs']
                )
            else:
                cooc_id = np.random.choice(self.all_item_ids)
                if self.call_count % 500 == 0 and idx < 5:
                    print(f"  idx={idx}: item_id={item_id_val} fallback random")
            if self.call_count % 500 == 0 and idx < 5:
                    print(f"  idx={idx}: item_id={item_id_val} cooc_id={cooc_id}")
            cooc_emb = self.item_id_to_embedding.get(cooc_id, batch['embedding'][0])
            cooccurrence_embeddings.append(cooc_emb)
        
        batch['cooccurrence_embedding'] = torch.stack(cooccurrence_embeddings)
        return batch
    
#запустить сасрек, леттер, sasrec << tiger < letter < plum


class AddWeightedCooccurrenceEmbeddingsVectorized:
    
    def __init__(
        self,
        cooccur_counts: Dict[int, Dict[int, int]],
        item_id_to_embedding: Dict[int, torch.Tensor],
        all_item_ids: List[int],
        device: torch.device,
        limit_neighbors: bool = True,
        max_neighbors: int = 256,
        seed: int = 42,
        verbose: bool = True
    ):
        self.device = device
        self.call_count = 0
        self.limit_neighbors = limit_neighbors
        self.max_neighbors = max_neighbors
        self.seed = seed
        self.verbose = verbose
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Initializing AddWeightedCooccurrenceEmbeddingsVectorized")
            print(f"{'='*80}")
            init_start = time.time()
        
        all_item_ids_sorted = sorted(all_item_ids)
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(all_item_ids_sorted)}
        self.idx_to_item_id = torch.tensor(all_item_ids_sorted, device=device, dtype=torch.long)
        
        if self.verbose:
            print(f"[INIT] Sorted {len(all_item_ids)} item IDs and created mappings")
        
        num_items = len(all_item_ids_sorted)
        embedding_dim = next(iter(item_id_to_embedding.values())).shape[0]
        
        if self.verbose:
            print(f"[INIT] Num items: {num_items}, Embedding dim: {embedding_dim}")
        
        self.embedding_matrix = torch.zeros(
            size=(num_items, embedding_dim),
            device=device,
            dtype=torch.float32,
            requires_grad=False
        )
        
        emb_load_start = time.time()
        for item_id, emb in item_id_to_embedding.items():
            idx = self.item_id_to_idx[item_id]
            if isinstance(emb, torch.Tensor):
                self.embedding_matrix[idx] = emb.to(device).detach()
            else:
                self.embedding_matrix[idx] = torch.tensor(emb, device=device, dtype=torch.float32)
        
        if self.verbose:
            emb_load_time = time.time() - emb_load_start
            print(f"[INIT] Loaded {len(item_id_to_embedding)} embeddings in {emb_load_time*1000:.2f}ms")
        
        self._build_cooccurrence_tables(cooccur_counts, num_items)
        
        if self.verbose:
            init_time = time.time() - init_start
            print(f"[INIT] Total initialization time: {init_time*1000:.2f}ms")
            print(f"{'='*80}\n")
    
    def _build_cooccurrence_tables(self, cooccur_counts: Dict, num_items: int):
        if self.verbose:
            build_start = time.time()
            print(f"\n[BUILD] Building cooccurrence tables...")
        
        indexed_cooccur_counts = {}
        for item_id, neighbors in cooccur_counts.items():
            if item_id in self.item_id_to_idx:
                idx = self.item_id_to_idx[item_id]
                indexed_neighbors = {}
                for neighbor_id, count in neighbors.items():
                    if neighbor_id in self.item_id_to_idx:
                        neighbor_idx = self.item_id_to_idx[neighbor_id]
                        indexed_neighbors[neighbor_idx] = count
                if indexed_neighbors:
                    indexed_cooccur_counts[idx] = indexed_neighbors
        
        if self.verbose:
            items_with_cooc = len(indexed_cooccur_counts)
            print(f"[BUILD] Items with cooccurrences: {items_with_cooc}/{num_items}")
            total_pairs = sum(len(neighbors) for neighbors in indexed_cooccur_counts.values())
            print(f"[BUILD] Total cooccurrence pairs: {total_pairs}")
        
        max_actual_neighbors = 0
        for idx in range(num_items):
            counter = indexed_cooccur_counts.get(idx)
            if counter and len(counter) > 0:
                num_neighbors = len(counter)
                if self.limit_neighbors:
                    num_neighbors = min(num_neighbors, self.max_neighbors)
            else:
                num_neighbors = num_items
            max_actual_neighbors = max(max_actual_neighbors, num_neighbors)
        
        if self.limit_neighbors:
            max_actual_neighbors = min(max_actual_neighbors, self.max_neighbors)
        
        if self.verbose:
            print(f"[BUILD] Max neighbors per item: {max_actual_neighbors}")
        
        neighbors_matrix = torch.zeros(
            (num_items, max_actual_neighbors),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        
        probs_matrix = torch.zeros(
            (num_items, max_actual_neighbors),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        
        valid_mask = torch.zeros(
            (num_items, max_actual_neighbors),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False
        )
        
        matrix_fill_start = time.time()
        
        for idx in range(num_items):
            counter = indexed_cooccur_counts.get(idx)
            
            if counter and len(counter) > 0:
                cooc_items = sorted(counter.items(), key=lambda x: x, reverse=True)
                cooc_ids, freqs = zip(*cooc_items)
                cooc_ids = list(cooc_ids)
                freqs = np.array(freqs, dtype=np.float32)
                
                num_neighbors = min(len(cooc_ids), max_actual_neighbors)
                cooc_ids = cooc_ids[:num_neighbors]
                freqs = freqs[:num_neighbors]
                
                probs = freqs / freqs.sum()
                
                neighbors_matrix[idx, :num_neighbors] = torch.tensor(
                    cooc_ids, dtype=torch.long, device=self.device
                )
                probs_matrix[idx, :num_neighbors] = torch.tensor(
                    probs, dtype=torch.float32, device=self.device
                )
                valid_mask[idx, :num_neighbors] = True
            
            else:
                if max_actual_neighbors >= num_items:
                    neighbors_matrix[idx, :num_items] = torch.arange(num_items, device=self.device)
                    probs_matrix[idx, :num_items] = 1.0 / num_items
                    valid_mask[idx, :num_items] = True
                else:
                    perm = torch.randperm(num_items, device=self.device)[:max_actual_neighbors]
                    neighbors_matrix[idx] = perm
                    probs_matrix[idx] = 1.0 / max_actual_neighbors
                    valid_mask[idx] = True
        
        if self.verbose:
            matrix_fill_time = time.time() - matrix_fill_start
            print(f"[BUILD] Filled matrices in {matrix_fill_time*1000:.2f}ms")
        
        self.neighbors_matrix = neighbors_matrix
        self.probs_matrix = probs_matrix
        self.valid_mask = valid_mask
        
        if self.verbose:
            print(f"[BUILD] neighbors_matrix shape: {neighbors_matrix.shape}")
            print(f"[BUILD] probs_matrix shape: {probs_matrix.shape}")
            print(f"[BUILD] valid_mask shape: {valid_mask.shape}")
            build_time = time.time() - build_start
            print(f"[BUILD] Total build time: {build_time*1000:.2f}ms")
    
    def __call__(self, batch):
        self.call_count += 1
        
        call_start = time.time()
        
        item_ids = batch['item_id']
        
        if not isinstance(item_ids, torch.Tensor):
            item_ids = torch.tensor(item_ids, device=self.device, dtype=torch.long)
        else:
            item_ids = item_ids.to(device=self.device, dtype=torch.long)
        
        batch_size = item_ids.shape
        
        indexed_item_ids = torch.tensor(
            [self.item_id_to_idx.get(int(iid.item()), 0) for iid in item_ids],
            device=self.device,
            dtype=torch.long
        )
        
        probs = self.probs_matrix[indexed_item_ids]
        mask = self.valid_mask[indexed_item_ids]
        
        masked_probs = probs.clone()
        masked_probs[~mask] = 0.0
        
        row_sums = masked_probs.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0
        masked_probs = masked_probs / row_sums
        
        neighbor_indices = torch.multinomial(masked_probs, num_samples=1, replacement=True)
        neighbor_indices = neighbor_indices.squeeze(1)
        
        cooc_indexed_ids = self.neighbors_matrix[indexed_item_ids, neighbor_indices]
        cooccurrence_embeddings = self.embedding_matrix[cooc_indexed_ids]
        
        batch['cooccurrence_embedding'] = cooccurrence_embeddings
        
        call_time = time.time() - call_start
        if self.verbose and self.call_count % 1000 == 0:
            print(f"Call #{self.call_count}: batch_size={batch_size}, {call_time*1000:.2f}ms")
        
        return batch