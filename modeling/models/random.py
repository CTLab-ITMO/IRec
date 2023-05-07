from models.base import BaseModel

import torch


class RandomModel(BaseModel, config_name='random'):

    def __init__(
            self,
            label_prefix,
            candidate_prefix,
            num_items
    ):
        self._label_prefix = label_prefix
        self._candidate_prefix = candidate_prefix
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            label_prefix=config['label_prefix'],
            candidate_prefix=config['candidate_prefix'],
            num_items=kwargs['num_items']
        )

    def __call__(self, inputs):
        if '{}.ids'.format(self._candidate_prefix) in inputs:
            candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)
            batch_size = candidate_lengths.shape[0]
            num_candidates = candidate_lengths[0]
            candidate_scores = torch.rand(batch_size, num_candidates)  # (batch_size, num_candidates)
        else:
            labels_lengths = inputs['{}.length'.format(self._label_prefix)]  # (batch_size)
            batch_size = labels_lengths.shape[0]
            candidate_scores = torch.rand(batch_size, self._num_items + 1)  # (batch_size, num_items)
            candidate_scores[:, 0] = -torch.inf  # zero (padding) token
            candidate_scores[:, self._num_items + 1:] = -torch.inf  # all not real items-related things

        return candidate_scores
