import numpy as np
import pickle
import torch
import torch.nn.functional as F
from typing import Dict, List
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
                if self.call_count % 500 == 0 and idx < 5:
                    print(f"  idx={idx}: item_id={item_id_val} fallback random")
            
            cooc_emb = self.item_id_to_embedding.get(cooc_id, batch['embedding'][0])
            cooccurrence_embeddings.append(cooc_emb)

        batch['cooccurrence_embedding'] = torch.stack(cooccurrence_embeddings)
        return batch



class AddWeightedCooccurrenceEmbeddingsCached:
    def __init__(self, cooccur_counts, item_id_to_embedding, all_item_ids):
        self.cooccur_counts = cooccur_counts
        self.item_id_to_embedding = item_id_to_embedding
        self.all_item_ids = all_item_ids
        self.call_count = 0

        self.cooc_probs_cache = {}
        self._precompute_probabilities()
    
    def _precompute_probabilities(self):
        for item_id, counter in self.cooccur_counts.items():
            if counter and len(counter) > 0:
                cooc_ids, freqs = zip(*counter.items())
                freqs_array = np.array(freqs, dtype=np.float32)
                probs = freqs_array / freqs_array.sum()
                self.cooc_probs_cache[item_id] = (cooc_ids, probs)
    
    def __call__(self, batch):
        self.call_count += 1
        item_ids = batch['item_id']
        cooccurrence_embeddings = []
        
        for idx, item_id in enumerate(item_ids):
            item_id_val = int(item_id.item()) if torch.is_tensor(item_id) else int(item_id)
            
            if item_id_val in self.cooc_probs_cache:
                cooc_ids, probs = self.cooc_probs_cache[item_id_val]
                cooc_id = np.random.choice(cooc_ids, p=probs)
            else:
                cooc_id = np.random.choice(self.all_item_ids)
                if self.call_count % 10 == 0 and idx < 5:
                    print(f"  idx={idx}: item_id={item_id_val} fallback random")
            
            cooc_emb = self.item_id_to_embedding.get(cooc_id, batch['embedding'][0])
            cooccurrence_embeddings.append(cooc_emb)

        batch['cooccurrence_embedding'] = torch.stack(cooccurrence_embeddings)
        return batch

class AddWeightedCooccurrenceEmbeddingsVectorized:

    def __init__(
        self,
        cooccur_counts: Dict[int, Dict[int, int]],
        item_id_to_embedding: Dict[int, torch.Tensor],
        all_item_ids: List[int],
        device: torch.device,
        limit_neighbors: bool = True,
        max_neighbors: int = 256
    ):
        """
            limit_neighbors: если True, ограничиваем до max_neighbors (для экономии памяти)
            max_neighbors: максимум соседей (используется только если limit_neighbors=True)
        """
        self.device = device
        self.call_count = 0
        self.limit_neighbors = limit_neighbors
        self.max_neighbors = max_neighbors
        
        max_item_id = max(item_id_to_embedding.keys())
        embedding_dim = next(iter(item_id_to_embedding.values())).shape[0]
        
        self.embedding_matrix = torch.zeros(
            (max_item_id + 1, embedding_dim),
            device=device,
            dtype=torch.float32,
            requires_grad=False
        )
        
        print("Building embedding matrix")
        for item_id, emb in item_id_to_embedding.items():
            if isinstance(emb, torch.Tensor):
                self.embedding_matrix[item_id] = emb.detach()
            else:
                self.embedding_matrix[item_id] = torch.tensor(emb, device=device, dtype=torch.float32)
        
        self.all_item_ids_tensor = torch.tensor(
            all_item_ids,
            device=device,
            dtype=torch.long,
            requires_grad=False
        )
        
        print("Building cooccurrence tables")
        self._build_cooccurrence_tables(cooccur_counts, max_item_id, len(all_item_ids))
    
    def _build_cooccurrence_tables(self, cooccur_counts: Dict, max_item_id: int, num_all_items: int):
        """
        - neighbors_matrix: [max_item_id+1, num_neighbors]
        - probs_matrix: [max_item_id+1, num_neighbors]
        Если у item_id нет соседей, neighbors и probs заполняются равномерно из all_items
        """
        neighbor_counts = {}
        for item_id in range(max_item_id + 1):
            counter = cooccur_counts.get(item_id)
            if counter and len(counter) > 0:
                num_neighbors = len(counter)
                if self.limit_neighbors:
                    num_neighbors = min(num_neighbors, self.max_neighbors)
            else:
                num_neighbors = num_all_items
            
            neighbor_counts[item_id] = num_neighbors
        
        max_num_neighbors = max(neighbor_counts.values())
        actual_max_neighbors = min(max_num_neighbors, self.max_neighbors) if self.limit_neighbors else max_num_neighbors
        
        print(f"Max neighbors per item: {actual_max_neighbors}")
        
        neighbors_matrix = torch.zeros(
            (max_item_id + 1, actual_max_neighbors),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        
        probs_matrix = torch.zeros(
            (max_item_id + 1, actual_max_neighbors),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        
        num_items_with_cooc = 0
        
        # Заполняем матрицы
        for item_id in range(max_item_id + 1):
            counter = cooccur_counts.get(item_id)
            
            if counter and len(counter) > 0:
                # === Есть соседи: используем реальные вероятности ===
                num_items_with_cooc += 1
                
                # Извлекаем соседей и их counts, сортируем по частоте
                cooc_ids, freqs = zip(*sorted(counter.items(), key=lambda x: x[1], reverse=True))
                cooc_ids = list(cooc_ids)
                freqs = np.array(freqs, dtype=np.float32)
                
                # Берем только топ
                num_neighbors = min(len(cooc_ids), actual_max_neighbors)
                cooc_ids = cooc_ids[:num_neighbors]
                freqs = freqs[:num_neighbors]
                
                # Нормализуем
                probs = freqs / freqs.sum()
                
                neighbors_matrix[item_id, :num_neighbors] = torch.tensor(
                    cooc_ids, dtype=torch.long, device=self.device
                )
                probs_matrix[item_id, :num_neighbors] = torch.tensor(
                    probs, dtype=torch.float32, device=self.device
                )
            
            else:
                # Нет соседей: равномерное распределение на all_items
                if actual_max_neighbors >= num_all_items:
                    # Можем поместить всех айтемов
                    neighbors_matrix[item_id, :num_all_items] = self.all_item_ids_tensor
                    probs_matrix[item_id, :num_all_items] = 1.0 / num_all_items
                else:
                    # Выбираем случайное подмножество
                    indices = torch.randperm(num_all_items, device=self.device)[:actual_max_neighbors]
                    neighbors_matrix[item_id] = self.all_item_ids_tensor[indices]
                    probs_matrix[item_id] = 1.0 / actual_max_neighbors
        
        self.neighbors_matrix = neighbors_matrix
        self.probs_matrix = probs_matrix
        
        print(f"Cooccurrence tables built: {num_items_with_cooc}/{max_item_id + 1} items have real neighbors")
    
    def __call__(self, batch):
        self.call_count += 1
        
        item_ids = batch['item_id']  # [batch_size]
        batch_size = item_ids.shape[0]
        
        # Берем вероятности для items в батче
        probs = self.probs_matrix[item_ids]  # [batch_size, max_neighbors]
        
        # Выбираем индекс соседа для каждого item
        # torch.multinomial: выбирает из max_neighbors категорий по вероятностям
        # Результат: [batch_size, 1] - индексы в диапазоне [0, max_neighbors)
        neighbor_indices = torch.multinomial(probs, num_samples=1, replacement=True)
        neighbor_indices = neighbor_indices.squeeze(1)  # [batch_size]
        
        # neighbors_matrix[item_ids, neighbor_indices] -> [batch_size]
        cooc_ids = self.neighbors_matrix[item_ids, neighbor_indices]
        
        # Lookup эмбеддингов
        cooccurrence_embeddings = self.embedding_matrix[cooc_ids]  # [batch_size, embedding_dim]
        
        batch['cooccurrence_embedding'] = cooccurrence_embeddings
        
        # if self.call_count % 500 == 0:
        #     print(
        #         f"Call #{self.call_count}: {batch_size} samples, "
        #         f"cooc_embeddings shape: {cooccurrence_embeddings.shape}"
        #     )
        
        return batch