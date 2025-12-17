import json
from collections import defaultdict, Counter
from data import InteractionsDatasetParquet
from collections import defaultdict, Counter


class CoocMappingDataset:
    def __init__(
            self,
            train_sampler,
            num_items,
            cooccur_counter_mapping=None
    ):
        self._train_sampler = train_sampler
        self._num_items = num_items
        self._cooccur_counter_mapping = cooccur_counter_mapping

    @classmethod
    def create(cls, inter_json_path, window_size):
        max_item_id = 0
        train_dataset = []

        with open(inter_json_path, 'r') as f:
            user_interactions = json.load(f)

        for user_id_str, item_ids in user_interactions.items():
            user_id = int(user_id_str)
            if item_ids:
                max_item_id = max(max_item_id, max(item_ids))
            if len(item_ids) >= 5:
                print(f'Core-5 dataset is used, user {user_id} has only {len(item_ids)} items')
            train_dataset.append({
                'user_ids': [user_id],
                'item_ids': item_ids[:-2],
            })


        cooccur_counter_mapping = cls.build_cooccur_counter_mapping(train_dataset, window_size=window_size)
        print(f'Computed window-based co-occurrence mapping for {len(cooccur_counter_mapping)} items but max_item_id is {max_item_id}')


        train_sampler = train_dataset


        return cls(
            train_sampler=train_sampler,
            num_items=max_item_id + 1,
            cooccur_counter_mapping=cooccur_counter_mapping
        )


    @classmethod
    def create_from_split_part(
            cls,
            train_inter_parquet_path,
            window_size,
    ):

        max_item_id = 0
        train_dataset = []


        train_interactions = InteractionsDatasetParquet(train_inter_parquet_path)

        actions_num = 0
        for session in train_interactions:
            user_id, item_ids = int(session['user_id']), session['item_ids']
            if item_ids.any():
                max_item_id = max(max_item_id, max(item_ids))
            actions_num += len(item_ids)
            train_dataset.append({
                'user_ids': [user_id],
                'item_ids': item_ids,
            })


        print(f'Train: {len(train_dataset)} users')
        print(f'Max item ID: {max_item_id}')
        print(f"Actions num: {actions_num}")


        cooccur_counter_mapping = cls.build_cooccur_counter_mapping(
            train_dataset,
            window_size=window_size
        )


        print(f'Computed window-based co-occurrence mapping for {len(cooccur_counter_mapping)} items')


        return cls(
            train_sampler=train_dataset,
            num_items=max_item_id + 1,
            cooccur_counter_mapping=cooccur_counter_mapping
        )



    @staticmethod
    def build_cooccur_counter_mapping(train_dataset, window_size):
        cooccur_counts = defaultdict(Counter)
        for session in train_dataset:
            items = session['item_ids']
            for i in range(len(items)):
                item_i = items[i]
                for j in range(max(0, i - window_size), min(len(items), i + window_size + 1)):
                    if i != j:
                        cooccur_counts[item_i][items[j]] += 1
        max_hist_len = max(len(counter) for counter in cooccur_counts.values()) if cooccur_counts else 0
        print(f"Max cooccurrence history length is {max_hist_len}")
        return cooccur_counts



    @property
    def cooccur_counter_mapping(self):
        return self._cooccur_counter_mapping