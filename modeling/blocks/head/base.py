from utils import MetaParent, get_activation_function

import torch
import torch.nn as nn


class BaseHead(metaclass=MetaParent):
    pass


class TorchHead(BaseHead, nn.Module):

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None):
        return super().create_from_config(config)


class IdentityHead(TorchHead, config_name='identity'):

    def __call__(self, inputs):
        return torch.tensor(0.0), inputs


class TrainTestHead(TorchHead, config_name='train/test'):

    def __init__(self, train_head, test_head):
        super().__init__()
        self._train_head = train_head
        self._test_head = test_head

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None):
        return cls(
            train_head=BaseHead.create_from_config(config["train"]),
            test_head=BaseHead.create_from_config(config["test"])
        )

    def forward(self, inputs):
        if self.training:  # train mode
            return self._train_head(inputs)
        else:  # eval mode
            return self._test_head(inputs)


class CompositeHead(TorchHead, config_name='composite'):

    def __init__(self, heads, output_prefix=None):
        super().__init__()
        self._heads = heads
        self._output_prefix = output_prefix

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None):
        return cls(
            heads=nn.ModuleList([
                BaseHead.create_from_config(head_cfg, num_users=num_users, num_items=num_items)
                for head_cfg in config['heads']
            ]),
            output_prefix=config.get('output_prefix', None)
        )

    def forward(self, inputs):
        total_loss = 0.0

        for head in self._heads:
            loss, inputs = head(inputs)
            total_loss += loss

        if self._output_prefix is not None:
            inputs[self._output_prefix] = total_loss.cpu().item()

        return total_loss, inputs


class BPRHead(TorchHead, config_name='bpr'):

    def __init__(
            self,
            positive_prefix,
            negative_prefix,
            output_prefix=None,
            use_regularization=False,
            activation='softplus'
    ):
        super().__init__()
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._output_prefix = output_prefix
        self._use_regularization = use_regularization
        self._activation = get_activation_function(activation)

    def forward(self, inputs):
        positive_scores = inputs[self._positive_prefix]  # (all_batch_items)
        negative_scores = inputs[self._negative_prefix]  # (all_batch_items)
        assert positive_scores.shape[0] == negative_scores.shape[0]

        loss = torch.mean(self._activation(negative_scores - positive_scores))  # (1)
        if self._use_regularization:
            pass  # TODO

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss, inputs


class CrossEntropyHead(TorchHead, config_name='ce'):

    def __init__(self, predictions_prefix, labels_prefix, output_prefix=None):
        super().__init__()
        self._pred_prefix = predictions_prefix
        self._labels_prefix = labels_prefix
        self._output_prefix = output_prefix

        self._loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix]  # (all_batch_items, num_classes)
        all_labels = inputs['{}.ids'.format(self._labels_prefix)].long()  # (all_batch_items)
        assert all_logits.shape[0] == all_labels.shape[0]

        loss = self._loss(all_logits, all_labels)
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss, inputs


class BinaryCrossEntropyHead(TorchHead, config_name='bce'):

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
        self._output_prefix = output_prefix

        if with_logits:
            self._loss = nn.BCEWithLogitsLoss()
        else:
            self._loss = nn.BCELoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix].float()  # (all_batch_items)
        all_labels = inputs[self._labels_prefix].float()  # (all_batch_items)
        assert all_logits.shape[0] == all_labels.shape[0]

        loss = self._loss(all_logits, all_labels)
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss, inputs
