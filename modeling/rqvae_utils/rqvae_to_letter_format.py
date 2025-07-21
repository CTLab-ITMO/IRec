import json
from pathlib import Path

import torch

from models import TigerModel
from utils import DEVICE


def convert_to_letter_format(
        semantic_ids: torch.Tensor,
        codebook_sizes: list[int],
        user_interactions_path: str,
        output_dir: str,
        dataset_name: str = "Beauty"
):
    """
    Конвертирует выход RQ-VAE и взаимодействия пользователей в формат LETTER

    Args:
        semantic_ids: семантические ID [num_items, num_codebooks]
        codebook_sizes: размеры кодбуков
        user_interactions_path: путь к файлу с взаимодействиями
        output_dir: директория для сохранения файлов
        dataset_name: название датасета (префикс файлов)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_items, num_codebooks = semantic_ids.shape
    semantic_ids_np = semantic_ids.numpy()

    # 1. Генерация {dataset_name}.index.json
    prefixes = [chr(ord('a') + i) for i in range(num_codebooks)]
    index_dict = {}

    for item_idx in range(num_items):
        tokens = []
        for codebook_idx in range(num_codebooks):
            code = semantic_ids_np[item_idx, codebook_idx]
            prefix = prefixes[codebook_idx]
            tokens.append(f"<{prefix}_{code}>")
        index_dict[str(item_idx)] = tokens

    # Сохраняем index.json
    index_path = output_path / f"{dataset_name}.index.json"
    with open(index_path, "w") as f:
        json.dump(index_dict, f, indent=2)

    # 2. Генерация {dataset_name}.inter.json
    inter_dict = {}

    # Читаем и преобразуем взаимодействия
    with open(user_interactions_path, "r") as f:
        for user_idx, line in enumerate(f):
            parts = line.strip().split()
            orig_item_ids = list(map(int, parts))

            item_indices = [str(item_id - 1) for item_id in orig_item_ids]
            inter_dict[str(user_idx)] = item_indices

    inter_path = output_path / f"{dataset_name}.inter.json"
    with open(inter_path, "w") as f:
        json.dump(inter_dict, f, indent=2)

    print(f"Файлы LETTER созданы:")
    print(f" - Семантические коды: {index_path}")
    print(f" - Взаимодействия: {inter_path}")


# Пример конфигурации для вызова
if __name__ == "__main__":
    # Пример конфигурации (должна приходить извне)

    BASE_DIR = Path(__file__).parent.parent.parent
    config = {
        "rqvae_train_config_path": f"{BASE_DIR}/configs/train/rqvae_train_config.json",
        "rqvae_checkpoint_path": f"{BASE_DIR}/checkpoints/rqvae_large_beauty_ddddddd_final_state.pth",
        "embs_extractor_path": f"{BASE_DIR}/data/Beauty/data_full.pt",

        "user_interactions_path": f"{BASE_DIR}/data/Beauty/all_data.txt",
        "output_dir": f"{BASE_DIR}/data/Beauty/letter_format",
        "dataset_name": "Beauty"
    }
    print(config)
    # Инициализация RQ-VAE с автоматическим сохранением в формате LETTER
    rqvae_model, semantic_ids, residuals, item_ids = TigerModel.init_rqvae(config)

    print("bb")
    convert_to_letter_format(
        semantic_ids=semantic_ids.cpu(),
        codebook_sizes=rqvae_model.codebook_sizes,
        user_interactions_path=config["user_interactions_path"],
        output_dir=config.get("output_dir", "."),
        dataset_name=config.get("dataset_name", "Beauty")
    )

