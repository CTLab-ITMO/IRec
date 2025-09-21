from .base import BaseModel

import torch


class PopModel(BaseModel, config_name='pop'):
    def __init__(self, label_prefix, counts_prefix, num_items):
        self._label_prefix = label_prefix
        self._counts_prefix = counts_prefix
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            label_prefix=config['label_prefix'],
            counts_prefix=config['counts_prefix'],
            num_items=kwargs['num_items'],
        )

    def __call__(self, inputs):
        candidate_counts = inputs[
            '{}.ids'.format(self._counts_prefix)
        ]  # (all_batch_candidates)
        candidate_counts_lengths = inputs[
            '{}.length'.format(self._counts_prefix)
        ]  # (batch_size)
        batch_size = candidate_counts_lengths.shape[0]

        candidate_scores = torch.reshape(
            candidate_counts,
            shape=(batch_size, self._num_items + 2),
        ).float()  # (batch_size, num_items)
        candidate_scores[:, 0] = -torch.inf  # zero (padding) token
        candidate_scores[
            :,
            self._num_items + 1 :,
        ] = -torch.inf  # all not real items-related things

        _, indices = torch.topk(
            candidate_scores,
            k=20,
            dim=-1,
            largest=True,
        )  # (batch_size, 20)

        return indices
