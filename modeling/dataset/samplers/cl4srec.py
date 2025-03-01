import copy

import numpy as np

from dataset.samplers.base import EvalSampler, TrainSampler


class Cl4SRecTrainSampler(TrainSampler, config_name="cl4srec"):
    def __init__(
        self,
        dataset,
        num_users,
        num_items,
        item_crop_portion,
        item_mask_portion,
        item_reorder_portion,
    ):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._mask_item_idx = self._num_items + 1
        self._item_crop_portion = item_crop_portion
        self._item_mask_portion = item_mask_portion
        self._item_reorder_portion = item_reorder_portion

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs["dataset"],
            num_users=kwargs["num_users"],
            num_items=kwargs["num_items"],
            item_crop_portion=config["item_crop_portion"],
            item_mask_portion=config["item_mask_portion"],
            item_reorder_portion=config["item_reorder_portion"],
        )

    def _apply_crop_augmentation(self, item_sequence):
        num_elements_to_crop = max(1, int(self._item_crop_portion * len(item_sequence)))
        crop_start_index = np.random.randint(
            low=0, high=len(item_sequence) - num_elements_to_crop + 1
        )
        assert 0 <= crop_start_index <= len(item_sequence) - num_elements_to_crop
        item_sequence = item_sequence[
            crop_start_index : crop_start_index + num_elements_to_crop
        ]
        return item_sequence

    def _apply_mask_augmentation(self, item_sequence):
        for idx in range(len(item_sequence)):
            p = np.random.uniform(low=0.0, high=1.0)
            if p < self._item_mask_portion:
                item_sequence[idx] = self._mask_item_idx

            if p < self._item_mask_portion:
                p /= self._item_mask_portion

                if p < 0.8:
                    item_sequence[idx] = self._mask_item_idx
                elif p < 0.9:
                    item_sequence[idx] = np.random.randint(1, self._num_items + 1)
                else:
                    pass  # item_sequence[idx] = item_sequence[idx]
            else:
                pass  # item_sequence[idx] = item_sequence[idx]

        return item_sequence

    def _apply_reorder_augmentation(self, item_sequence):
        num_elements_to_reorder = int(self._item_reorder_portion * len(item_sequence))
        reorder_start_index = np.random.randint(
            low=0, high=len(item_sequence) - num_elements_to_reorder + 1
        )
        assert 0 <= reorder_start_index <= len(item_sequence) - num_elements_to_reorder
        reordered_subsequence = item_sequence[
            reorder_start_index : reorder_start_index + num_elements_to_reorder
        ]
        np.random.shuffle(reordered_subsequence)  # This works in-place
        item_sequence = (
            item_sequence[:reorder_start_index]
            + reordered_subsequence
            + item_sequence[reorder_start_index + num_elements_to_reorder :]
        )
        return item_sequence

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample["item.ids"][:-1]
        next_item = sample["item.ids"][-1]

        sample_items = set(sample["item.ids"])
        negative = np.random.randint(low=1, high=self._num_items + 1)
        while negative in sample_items:
            negative = np.random.randint(low=1, high=self._num_items + 1)

        augmentation_list = [
            self._apply_crop_augmentation,
            self._apply_mask_augmentation,
            self._apply_reorder_augmentation,
        ]

        fst_augmentation = np.random.choice(augmentation_list)
        snd_augmentation = np.random.choice(augmentation_list)

        fst_augmented_sequence = fst_augmentation(item_sequence)
        snd_augmented_sequence = snd_augmentation(item_sequence)

        return {
            "user.ids": sample["user.ids"],
            "user.length": sample["user.length"],
            "item.ids": item_sequence,
            "item.length": len(item_sequence),
            "fst_augmented_item.ids": fst_augmented_sequence,
            "fst_augmented_item.length": len(fst_augmented_sequence),
            "snd_augmented_item.ids": snd_augmented_sequence,
            "snd_augmented_item.length": len(snd_augmented_sequence),
            "labels.ids": [next_item],
            "labels.length": 1,
            "positive.ids": [next_item],
            "positive.length": 1,
            "negative.ids": [negative],
            "negative.length": 1,
        }


class Cl4SRecEvalSampler(EvalSampler, config_name="cl4srec"):
    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs["dataset"],
            num_users=kwargs["num_users"],
            num_items=kwargs["num_items"],
        )
