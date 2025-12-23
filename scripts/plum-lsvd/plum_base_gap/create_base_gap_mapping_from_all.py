import json
import pandas as pd
from pathlib import Path


ALL_MAPPING_PATH = "/home/jovyan/IRec/results-lsvd-2/base_gap/all_items_plum_vk-lsvd-15ts_base_with_gap_cb_512_ws_2_k_3000_e35_con_0.01_rqvae_1.0_clusters_colisionless.json"
TRAIN_INTERACTIONS_PATH = "/home/jovyan/IRec/sigir/lsvd_data_filtered/15-ts-ows/base_with_gap_interactions_grouped.parquet"
OUTPUT_TRAIN_MAPPING_PATH = "/home/jovyan/IRec/results-lsvd-2/base_gap/only_base_with_gap_plum_vk-lsvd-15ts_base_with_gap_cb_512_ws_2_k_3000_e35_con_0.01_rqvae_1.0_clusters_colisionless_from_all.json"


with open(ALL_MAPPING_PATH, 'r') as f:
    all_mapping = json.load(f)
print(f"Loaded {len(all_mapping)} items from all_mapping")

train_interactions = pd.read_parquet(TRAIN_INTERACTIONS_PATH)

train_item_ids = set()
for item_ids_array in train_interactions['item_ids']:
    train_item_ids.update(item_ids_array)

print(f"Found {len(train_item_ids)} unique train items")

train_mapping = {}
missing_count = 0

for item_id in train_item_ids:
    item_id_str = str(item_id)
    if item_id_str in all_mapping:
        train_mapping[item_id_str] = all_mapping[item_id_str]
    else:
        missing_count += 1

if missing_count > 0:
    print(f"{missing_count} items from train not found in all_mapping")

print(f"Created train_mapping with {len(train_mapping)} items")

Path(OUTPUT_TRAIN_MAPPING_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_TRAIN_MAPPING_PATH, 'w') as f:
    json.dump(train_mapping, f, indent=2)
print(f"Saved to {OUTPUT_TRAIN_MAPPING_PATH}")

print(f"all_mapping size: {len(all_mapping)}")
print(f"train_mapping size: {len(train_mapping)}")
print(f"train_mapping/all_mapping ratio: {len(train_mapping)/len(all_mapping):.1%}")

sample_matches = 0
for item_id_str in list(train_mapping.keys())[:100]:
    if all_mapping[item_id_str] == train_mapping[item_id_str]:
        sample_matches += 1

print(f"Verified: {sample_matches}/100 sampled items have identical codes")
