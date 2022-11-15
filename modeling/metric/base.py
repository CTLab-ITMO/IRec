from utils import MetaParent

import torch


class BaseMetric(metaclass=MetaParent):
    pass


class StaticMetric(BaseMetric, config_name='static'):
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def __call__(self, inputs):
        inputs[self._name] = self._value

        return inputs


class CompositeMetric(BaseMetric, config_name='composite'):

    def __init__(self, metrics):
        self._metrics = metrics

    @classmethod
    def create_from_config(cls, config):
        return cls(
            metrics=[BaseMetric.create_from_config(cfg) for cfg in config['metrics']]
        )

    def __call__(self, inputs):
        for metric in self._metrics:
            inputs = metric(inputs)
        return inputs


class NDCGMetric(BaseMetric, config_name='ndcg'):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix]  # (batch_size, num_candidates)
        predictions = (-predictions).argsort(dim=1)  # (batch_size, num_candidates)
        predictions = predictions[:, :self._k]  # (batch_size, k)

        labels = inputs['{}.ids'.format(labels_prefix)].float()  # (batch_size, num_candidates)
        answer_count = labels.sum(dim=1)  # (batch_size)

        hits = labels.gather(dim=1, index=predictions)  # (batch_size, k)

        position = torch.arange(2, 2 + self._k)  # (k)
        weights = 1 / torch.log2(position.float())  # (k)
        dcg = (hits * weights.to(hits.device)).sum(dim=1)  # (batch_size)
        idcg = torch.Tensor([weights[:min(int(n), self._k)].sum() for n in answer_count]).to(dcg.device)  # (batch_size)
        ndcg = (dcg / idcg).mean()  # (1)

        return ndcg.cpu().item()


class RecallMetric(BaseMetric, config_name='recall'):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix]  # (batch_size, num_candidates)
        predictions = (-predictions).argsort(dim=1)  # (batch_size, num_candidates)
        predictions = predictions[:, :self._k]  # (batch_size, k)

        labels = inputs['{}.ids'.format(labels_prefix)].float()  # (batch_size, num_candidates)
        hits = labels.gather(dim=1, index=predictions)  # (batch_size, k)

        recall = (
                hits.sum(dim=1) /
                torch.min(torch.Tensor([self._k]).to(labels.device), labels.sum(dim=1).float())
        ).mean()  # (1)

        return recall.cpu().item()
