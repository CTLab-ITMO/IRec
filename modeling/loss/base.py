import torch.nn

from utils import MetaParent

import torch.nn as nn


class BaseLoss(metaclass=MetaParent):
    pass


class TorchLoss(BaseLoss, nn.Module):
    pass


class IdentityLoss(BaseLoss, config_name='identity'):

    def __call__(self, inputs):
        return inputs


class CompositeLoss(TorchLoss, config_name='composite'):

    def __init__(self, losses, weights=None, output_prefix=None):
        super().__init__()
        self._losses = losses
        self._weights = weights or [1] * len(losses)
        self._output_prefix = output_prefix

    @classmethod
    def create_from_config(cls, config):
        return cls(
            losses=[BaseLoss.create_from_config(cfg) for cfg in config['losses']],
            weights=config.get('weights', [1] * len(config['losses'])),
            output_prefix=config.get('output_prefix', None)
        )

    def forward(self, inputs):
        total_loss = 0.0
        for loss, weight in zip(self._losses, self._weights):
            total_loss += weight * loss(inputs)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = total_loss.cpu().item()

        return total_loss


class CrossEntropyLoss(TorchLoss, config_name='ce'):

    def __init__(self, predictions_prefix, labels_prefix, output_prefix=None):
        super().__init__()
        self._pred_prefix = predictions_prefix
        self._labels_prefix = labels_prefix
        self._output_prefix = output_prefix

        self._loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix]  # (all_items, num_classes)
        all_labels = inputs['{}.ids'.format(self._labels_prefix)]  # (all_items)
        assert all_logits.shape[0] == all_labels.shape[0]

        loss = self._loss(all_logits, all_labels)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class BinaryCrossEntropyLoss(TorchLoss, config_name='bce'):

    def __init__(
            self,
            predictions_prefix,
            labels_prefix,
            with_logits=True,
            output_prefix=None
    ):
        super().__init__()
        self._pred_prefix = predictions_prefix
        self._labels_prefix = labels_prefix

        if with_logits:
            self._loss = nn.BCEWithLogitsLoss()
        else:
            self._loss = nn.BCELoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix]  # (all_items)
        all_labels = inputs[self._labels_prefix]  # (all_items)
        assert all_logits.shape[0] == all_labels.shape[0]
        loss = self._loss(all_logits, all_labels)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss

# TODO add parameters penalty for sasrec loss
