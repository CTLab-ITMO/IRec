import copy
import random

from dataset.samplers.base import EvalSampler, TrainSampler


class DuorecTrainSampler(TrainSampler, config_name="duorec"):
    def __init__(self, dataset, num_users, num_items):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs["dataset"],
            num_users=kwargs["num_users"],
            num_items=kwargs["num_items"],
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample["item.ids"]

        target_item = item_sequence[-1]
        item_sequence = item_sequence[:-1]

        # There is a probability of sampling the same sequence
        semantic_similar_sequence = random.choice(self._target_2_sequences[target_item])

        return {
            "user.ids": sample["user.ids"],
            "user.length": sample["user.length"],
            "item.ids": item_sequence,
            "item.length": len(item_sequence),
            "labels.ids": [target_item],
            "labels.length": 1,
            "semantic_similar_item.ids": semantic_similar_sequence,
            "semantic_similar_item.length": len(semantic_similar_sequence),
        }


class DuoRecEvalSampler(EvalSampler, config_name="duorec"):
    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs["dataset"],
            num_users=kwargs["num_users"],
            num_items=kwargs["num_items"],
        )
