from utils import MetaParent

import torch


class BaseMetric(metaclass=MetaParent):
    pass


class StaticMetric(BaseMetric, config_name='dummy'):
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
        return cls(metrics=[
            BaseMetric.create_from_config(cfg)
            for cfg in config['metrics']
        ])

    def __call__(self, inputs):
        for metric in self._metrics:
            inputs = metric(inputs)
        return inputs


class NDCGMetric(BaseMetric, config_name='ndcg'):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix]  # (batch_size, num_candidates)
        labels = inputs['{}.ids'.format(labels_prefix)]  # (all_batch_items) or (batch_size)

        if labels.shape[0] == predictions.shape[0]:
            new_labels = torch.zeros_like(predictions)  # (batch_size, num_candidates)
            labels = labels.unsqueeze(1)  # (batch_size, 1)
            labels = new_labels.scatter(
                src=torch.ones_like(labels).float(),
                dim=1,
                index=labels.long()
            ).float()  # (batch_size, num_candidates)
        else:
            labels = torch.reshape(labels, predictions.shape).float()  # (batch_size, num_candidates)

        predictions = (-predictions).argsort(dim=-1)  # (batch_size, num_candidates)
        predictions = predictions[..., :self._k]  # (batch_size, k)
        hits = labels.gather(dim=-1, index=predictions)  # (batch_size, k)

        answer_count = labels.sum(dim=-1)  # (batch_size)
        discount_factor = 1 / torch.log2(torch.arange(1, self._k + 1, 1).float() + 1.).to(hits.device)  # (k)

        dcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size)
        idcg = torch.Tensor([
            discount_factor[:min(int(n), self._k)].sum() for n in answer_count
        ]).to(dcg.device)  # (batch_size)
        ndcg = (dcg / idcg)  # (batch_size)

        return ndcg.cpu().tolist()


class RecallMetric(BaseMetric, config_name='recall'):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix]  # (batch_size, num_candidates)
        labels = inputs['{}.ids'.format(labels_prefix)]  # (all_batch_items) or (batch_size)

        if labels.shape[0] == predictions.shape[0]:
            new_labels = torch.zeros_like(predictions)  # (batch_size, num_candidates)
            labels = labels.unsqueeze(1)  # (batch_size, 1)
            labels = new_labels.scatter(
                src=torch.ones_like(labels).float(),
                dim=1,
                index=labels.long()
            ).float()  # (batch_size, num_candidates)
        else:
            labels = torch.reshape(labels, predictions.shape).float()  # (batch_size, num_candidates)

        predictions = (-predictions).argsort(dim=-1)  # (batch_size, num_candidates)
        predictions = predictions[..., :self._k]  # (batch_size, k)
        hits = labels.gather(dim=-1, index=predictions)  # (batch_size, k)

        recall = (
                hits.sum(dim=-1) / torch.minimum(labels.sum(dim=-1).float(), labels.new_ones(predictions.shape[0]).float() * self._k)
        )  # (batch_size)

        return recall.cpu().tolist()
