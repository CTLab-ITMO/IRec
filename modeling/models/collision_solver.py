from collections import defaultdict
from typing import List, Tuple, Dict
import torch

class CollisionSolver:
    def __init__(self, residual_length, semantic_id_length, device: torch.device = torch.device('cpu')):
        """
        :param residual_length: Длина остатка для каждого semantic_id
        :param semantic_id_length: Длина semantic_id (без токена решающего коллизии)
        :param device: Устройство
        """
        self._semantic_id_dict = defaultdict(list)
        self.residual_length = residual_length
        self.semantic_id_length = semantic_id_length
        self.device = device

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Перенос тензора на устройство
        """
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        return tensor

    def add_item(self, semantic_id: List[int] | torch.Tensor, residual: torch.Tensor) -> None:
        """
        Добавляет новый элемент в словарь хранящий semantic_ids с остатками

        :param semantic_id: Semantic id (без токена решающего коллизии)
        :param residual: Тензор с остатком для данного semantic_id
        """
        if isinstance(semantic_id, torch.Tensor):
            semantic_id = semantic_id.tolist()

        assert isinstance(residual, torch.Tensor)
        assert residual.shape == (self.residual_length,)
        assert len(semantic_id) == self.semantic_id_length

        residual = self._to_device(residual)
        key = tuple(semantic_id)
        self._semantic_id_dict[key].append((len(self._semantic_id_dict[key]), residual))


    def create_query_candidates_dict(self, semantic_ids: torch.Tensor | List[List[int]], residuals: torch.Tensor | List[List[int]]) -> None:
        """
        Создает словарь, который содержит сгруппирированные по semantic id элементы, к ним добавлены токены решающие коллизии (добавляются по порядку начиная с нуля)

        :param semantic_ids: Тензор или список всех semantic_id, полученных из rq-vae (без токенов решающих коллизии)
        :param residuals: Тензор или список остатков для каждого semantic_id
        """
        residuals_count = residuals.shape[0] if isinstance(residuals, torch.Tensor) else len(residuals)
        semantic_ids_count = semantic_ids.shape[0] if isinstance(semantic_ids, torch.Tensor) else len(semantic_ids)
        assert(residuals_count == semantic_ids_count)

        if isinstance(residuals, list):
            residuals = torch.tensor(residuals, device=self.device)
        residuals = self._to_device(residuals)

        for semantic_id, residual in zip(semantic_ids, residuals):
            self.add_item(semantic_id, residual)

    def get_candidates_tensor(self, query_prefixes: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param query_prefixes: [num_prefixes, prefix_len] список из semantic id (без токенов решающих коллизии)

        :return: Кортеж из двух тензоров:
        - candidates_tensor (размерность: [num_prefixes, max_collisions, residual_dim]): тензор, содержащий остатки кандидатов для каждого префикса
          `max_collisions` — максимальное количество кандидатов для каждого префикса
        - mask (размерность: [num_prefixes, max_collisions]): Маска для candidates_tensor

        Примечание:
            Предполагаем что все префиксы из `query_prefixes` уже есть в словаре semantic ids
            Если префикс не найден, будет выброшено исключение
        """
        assert isinstance(query_prefixes, list)
        assert(self.residual_length == len(self._semantic_id_dict[tuple(query_prefixes[0])][0][1]))
        assert(len(query_prefixes[0]) == self.semantic_id_length)

        max_collision_len = max(len(x) for x in self._semantic_id_dict.values())
        candidates_tensor = torch.zeros(len(query_prefixes), max_collision_len, self.residual_length, dtype=torch.float32, device=self.device)
        mask = torch.zeros(len(query_prefixes), max_collision_len, dtype=torch.bool, device=self.device)

        for i, semantic_id in enumerate(query_prefixes):
            key = tuple(semantic_id)
            assert key in self._semantic_id_dict.keys(), f"Не найдено обьектов с semantic id {key}" # нужно что-то с этим делать
            for j, residual in self._semantic_id_dict[key]: #сохранение порядка
                candidates_tensor[i, j] = residual
                mask[i, j] = True
        return candidates_tensor, mask

    def get_semantic_ids(self, query_prefixes: torch.Tensor, query_residuals: torch.Tensor) -> torch.Tensor:
        """
        :param query_prefixes: [num_prefixes, prefix_len] список из semantic id (без токенов решающих коллизии)

        :return: semantic_ids: [num_prefixes, prefix_len + 1] список из semantic id с токенами решающие коллизии
        """
        assert isinstance(query_prefixes, torch.Tensor)
        assert isinstance(query_residuals, torch.Tensor)
        assert(query_prefixes.shape[0] == query_residuals.shape[0])
        assert(query_prefixes.shape[1] == self.semantic_id_length)
        assert(query_residuals.shape[1] == self.residual_length)

        query_prefixes = self._to_device(query_prefixes)
        query_residuals = self._to_device(query_residuals)

        candidates_tensor, mask = self.get_candidates_tensor(query_prefixes.tolist())

        masked_dot_products = torch.einsum('ijk,ik->ij', candidates_tensor, query_residuals).masked_fill(~mask, float('-inf'))
        max_indices = torch.argmax(masked_dot_products, dim=1)
        best_semantic_ids = torch.concat((query_prefixes, max_indices.unsqueeze(1)), dim=1)
        return best_semantic_ids
    
    def get_closest_torch(self, query_prefixes: torch.Tensor, query_residuals: torch.Tensor):
        raise NotImplementedError("get_closest_torch is not implemented")
    
        query_prefixes = query_prefixes.to(self.device)
        query_residuals = query_residuals.to(self.device)

        batch_size, max_length = self._semantic_ids.shape
        num_prefixes, prefix_len = query_prefixes.shape

        # привожу к одной размерности чтобы найти совпадения по префиксам
        semantic_ids_exp = self._semantic_ids[:, :prefix_len].unsqueeze(0).expand(num_prefixes, batch_size, prefix_len) # [num_prefixes, batch_size, prefix_len]
        prefixes_exp = query_prefixes.unsqueeze(1).expand(num_prefixes, batch_size, prefix_len) #torch.tile
        is_prefix_match = (semantic_ids_exp == prefixes_exp).all(dim=2)  # [num_prefixes, batch_size]

        # Шаг 2: Маскирование residuals для каждого префикса
        residuals_exp = self._residuals.unsqueeze(0).expand(num_prefixes, batch_size, -1)  # [num_prefixes, batch_size, emb_dim]
        masked_residuals = residuals_exp * is_prefix_match.unsqueeze(2).float()  # Зануляем строки, не соответствующие префиксам
        dot_products = torch.einsum('ijk,ik->ij', masked_residuals, query_residuals)
        max_indices = torch.argmax(dot_products, dim=1)  # [num_prefixes] #

        best_semantic_ids = self._semantic_ids[max_indices]  # [num_prefixes, max_length]
        best_residuals = self._residuals[max_indices]  # [num_prefixes, emb_dim]
        
        return best_semantic_ids, best_residuals