import copy

from utils import MetaParent, get_activation_function, maybe_to_list, DEVICE

import torch
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
        self._weights = weights or [1.0] * len(losses)
        self._output_prefix = output_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        losses = []
        weights = []

        for loss_cfg in copy.deepcopy(config)['losses']:
            weight = loss_cfg.pop('weight') if 'weight' in loss_cfg else 1.0
            loss_function = BaseLoss.create_from_config(loss_cfg)

            weights.append(weight)
            losses.append(loss_function)

        return cls(losses=losses, weights=weights, output_prefix=config.get('output_prefix'))

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

        loss = self._loss(all_logits, all_labels)  # (1)
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
        self._output_prefix = output_prefix

        if with_logits:
            self._loss = nn.BCEWithLogitsLoss()
        else:
            self._loss = nn.BCELoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix].float()  # (all_batch_items)
        all_labels = inputs[self._labels_prefix].float()  # (all_batch_items)
        assert all_logits.shape[0] == all_labels.shape[0]

        loss = self._loss(all_logits, all_labels)  # (1)
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class BPRLoss(TorchLoss, config_name='bpr'):

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

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class RegularizationLoss(TorchLoss, config_name='regularization_loss'):

    def __init__(self, prefix, output_prefix=None):
        super().__init__()
        self._prefix = maybe_to_list(prefix)
        self._output_prefix = output_prefix

    def forward(self, inputs):
        loss = 0.0
        for prefix in self._prefix:
            loss += inputs[prefix].norm(2).pow(2)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class FpsLoss(TorchLoss, config_name='fps'):

    def __init__(
            self,
            fst_embeddings_prefix,
            snd_embeddings_prefix,
            tau=1.0,
            add_negatives=False,
            normalize_embeddings=False,
            output_prefix=None
    ):
        super().__init__()
        self._fst_embeddings_prefix = fst_embeddings_prefix
        self._snd_embeddings_prefix = snd_embeddings_prefix
        self._tau = tau
        self._add_negatives = add_negatives
        self._normalize_embeddings = normalize_embeddings
        self._output_prefix = output_prefix

    def forward(self, inputs):
        fst_embeddings = inputs[self._fst_embeddings_prefix]  # (x, embedding_dim)
        snd_embeddings = inputs[self._snd_embeddings_prefix]  # (x, embedding_dim)

        if self._normalize_embeddings:
            fst_embeddings = torch.nn.functional.normalize(fst_embeddings, dim=1)  # (x, embedding_dim)
            snd_embeddings = torch.nn.functional.normalize(snd_embeddings, dim=1)  # (x, embedding_dim)

        similarity_matrix = torch.matmul(fst_embeddings, snd_embeddings.T)  # (x, x)
        similarity_matrix = torch.exp(similarity_matrix / self._tau)  # (x, x)

        num_samples = similarity_matrix.shape[0]
        mask = torch.eye(num_samples, dtype=torch.bool).to(DEVICE)  # (x, x)

        positive_score = similarity_matrix[mask]  # (x)
        negative_score = torch.sum(similarity_matrix[~mask].reshape(num_samples, num_samples - 1), dim=-1)  # (x)

        if self._add_negatives:
            identity_similarity_matrix = torch.matmul(fst_embeddings, fst_embeddings.T)  # (x, x)
            identity_similarity_matrix = torch.exp(identity_similarity_matrix / self._tau)  # (x, x)

            negative_score += torch.sum(
                identity_similarity_matrix[~mask].reshape(num_samples, num_samples - 1),
                dim=-1
            )  # (x)

        loss = torch.mean(-torch.log(positive_score / (positive_score + negative_score)))  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class DuorecLoss(TorchLoss, config_name='duorec_ssl'):

    def __init__(
            self,
            fst_embeddings_prefix,
            snd_embeddings_prefix,
            normalized=False,
            tau=1.0,
            output_prefix=None
    ):
        super().__init__()
        self._fst_embeddings_prefix = fst_embeddings_prefix
        self._snd_embeddings_prefix = snd_embeddings_prefix
        self._normalized = normalized
        self._tau = tau
        self._output_prefix = output_prefix
        self._loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs):
        fst_embeddings = inputs[self._fst_embeddings_prefix]  # (x, embedding_dim)
        snd_embeddings = inputs[self._snd_embeddings_prefix]  # (x, embedding_dim)

        if self._normalized:
            fst_norms = torch.norm(fst_embeddings, p=2, dim=1)  # (x)
            snd_norms = torch.norm(snd_embeddings, p=2, dim=1)  # (x)
            fst_embeddings /= fst_norms
            snd_embeddings /= snd_norms

        similarity_matrix = torch.matmul(fst_embeddings, snd_embeddings.T)  # (x, x)
        similarity_matrix /= self._tau  # (x, x)
        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(DEVICE)  # (x, x)

        logits = similarity_matrix.reshape(-1).float()  # (x^2)
        labels = mask.reshape(-1).float()  # (x^2)

        loss = self._loss_func(logits, labels)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss
