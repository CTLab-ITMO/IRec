from dataset.samplers.base import TrainSampler, EvalSampler

import copy

from dataset.samplers.base import EvalSampler, TrainSampler
from dataset.negative_samplers.base import BaseNegativeSampler

class LastItemPredictionTrainSampler(TrainSampler, config_name='last_item_prediction'):

class LastItemPredictionTrainSampler(TrainSampler, config_name="last_item_prediction"):
    def __init__(self, dataset, num_users, num_items, negative_sampler, num_negatives):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._num_negatives = num_negatives

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config(
            {"type": config["negative_sampler_type"]}, **kwargs
        )

        return cls(
            dataset=kwargs["dataset"],
            num_users=kwargs["num_users"],
            num_items=kwargs["num_items"],
            negative_sampler=negative_sampler,
            num_negatives=config.get("num_negatives_train", 0),
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]
        last_item = sample['item.ids'][-1]

        if self._num_negatives == 0:
            return {
                "user.ids": sample["user.ids"],
                "user.length": sample["user.length"],
                "item.ids": item_sequence,
                "item.length": len(item_sequence),
                "labels.ids": [last_item],
                "labels.length": 1,
            }
        else:
            negative_sequence = self._negative_sampler.generate_negative_samples(
                sample, self._num_negatives
            )

            return {
                "user.ids": sample["user.ids"],
                "user.length": sample["user.length"],
                "item.ids": item_sequence,
                "item.length": len(item_sequence),
                "labels.ids": [last_item],
                "labels.length": 1,
                "negative.ids": negative_sequence,
                "negative.length": len(negative_sequence),
            }


class LastItemPredictionEvalSampler(EvalSampler, config_name='last_item_prediction'):

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )
