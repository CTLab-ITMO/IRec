from utils import MetaParent

import torch.nn as nn


class BaseLoss(metaclass=MetaParent):
    pass


class TorchLoss(BaseLoss, nn.Module):
    pass


class IdentityLoss(BaseLoss, config_name='identity'):

    def __call__(self, inputs):
        return inputs


class CrossEntropyLoss(TorchLoss, config_name='ce'):

    def __init__(self, predictions_prefix, labels_prefix, output_prefix):
        super().__init__()
        self._pred_prefix = predictions_prefix
        self._labels_prefix = labels_prefix
        self._output_prefix = output_prefix

        self._loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix]  # (all_items, num_classes)
        all_labels = inputs['{}.ids'.format(self._labels_prefix)]  # (all_items)
        assert all_logits.shape[0] == all_labels.shape[0]
        inputs[self._output_prefix] = self._loss(all_logits, all_labels)
        return inputs
