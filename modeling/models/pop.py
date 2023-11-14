from models.base import BaseModel

import torch


class PopModel(BaseModel, config_name='pop'):

    def __init__(
            self,
            label_prefix,
            candidate_prefix,
            counts_prefix,
            num_items
    ):
        self._label_prefix = label_prefix
        self._candidate_prefix = candidate_prefix
        self._counts_prefix = counts_prefix
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            label_prefix=config['label_prefix'],
            candidate_prefix=config['candidate_prefix'],
            counts_prefix=config['counts_prefix'],
            num_items=kwargs['num_items']
        )

    def __call__(self, inputs):
        candidate_counts = inputs['{}.ids'.format(self._counts_prefix)]  # (all_batch_candidates)
        candidate_counts_lengths = inputs['{}.length'.format(self._counts_prefix)]  # (batch_size)

        batch_size = candidate_counts_lengths.shape[0]
        num_candidates = candidate_counts_lengths[0]

        if '{}.ids'.format(self._candidate_prefix) in inputs:
            candidate_scores = torch.reshape(
                candidate_counts,
                shape=(batch_size, num_candidates)
            ).float()  # (batch_size, num_candidates)
        else:
            candidate_scores = torch.reshape(
                candidate_counts,
                shape=(batch_size, self._num_items + 1)
            ).float()  # (batch_size, num_items)
            candidate_scores[:, 0] = -torch.inf  # zero (padding) token
            candidate_scores[:, self._num_items + 1:] = -torch.inf  # all not real items-related things

        return candidate_scores  # (batch_size, num_candidates) / (batch_size, num_items)
