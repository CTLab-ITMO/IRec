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
        predictions = inputs[pred_prefix]  # (batch_size, all_items)
        positive = torch.unsqueeze(inputs['{}.ids'.format(labels_prefix)], dim=-1)  # (batch_size)

        labels = torch.zeros_like(predictions).scatter_(
            dim=1,
            index=positive,
            src=torch.ones_like(positive).float()
        )  # (batch_size, all_items)

        predictions = (-predictions).argsort(dim=-1)  # (batch_size, all_items)
        predictions = predictions[..., :self._k]  # (batch_size, k)
        hits = labels.gather(dim=-1, index=predictions)  # (batch_size, k)

        answer_count = labels.sum(dim=-1)  # (batch_size)
        discount_factor = 1 / torch.log2(torch.arange(1, self._k + 1, 1).float() + 1.).to(hits.device)  # (k)

        dcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size)
        idcg = torch.Tensor([
            discount_factor[:min(int(n), self._k)].sum() for n in answer_count
        ]).to(dcg.device)  # (batch_size)
        ndcg = (dcg / idcg)  # (batch_size)

        return ndcg.mean().cpu().item()


class RecallMetric(BaseMetric, config_name='recall'):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix]  # (batch_size, all_items)

        positive = torch.unsqueeze(inputs['{}.ids'.format(labels_prefix)], dim=-1)  # (batch_size)

        labels = torch.zeros_like(predictions).scatter_(
            dim=1,
            index=positive,
            src=torch.ones_like(positive).float()
        )  # (batch_size, all_items)

        predictions = (-predictions).argsort(dim=-1)  # (batch_size, all_items)
        predictions = predictions[..., :self._k]  # (batch_size, k)
        hits = labels.gather(dim=-1, index=predictions)  # (batch_size, k)

        recall = (
                hits.sum(dim=-1) / torch.minimum(labels.sum(dim=-1).float(), labels.new_ones(predictions.shape[0]).float() * self._k)
        )  # (batch_sie)

        return recall.mean().cpu().item()


class MRRMetric(BaseMetric, config_name='mrr'):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix]  # (batch_size, num_candidates)
        predictions = torch.argsort(predictions, dim=-1, descending=True)  # (batch_size, num_candidates)
        predictions = predictions[..., :self._k]  # (batch_size, k)
        predictions = torch.reshape(predictions, (predictions.shape[0], self._k))

        labels = inputs['{}.ids'.format(labels_prefix)].float()  # (all_batch_items)
        labels = torch.reshape(labels, (predictions.shape[0], -1))  # (batch_size, num_candidates)
        hits = labels.gather(dim=1, index=predictions)  # (batch_size, k)

        good_samples = hits[hits.sum(dim=-1) > 0]  # (good_samples_num, k)
        idx = reversed(torch.Tensor(range(1, self._k + 1)))  # (k)
        idx = torch.einsum("gk,k->gk", (good_samples, idx))  # (good_samples_num, k)
        indices = torch.argmax(idx, dim=-1)  # (good_samples_num)

        result = predictions.new_zeros((predictions.shape[0]))
        result[hits.sum(dim=-1) > 0] = 1.0 / (indices + 1.0)
        mrr = result.mean()

        return mrr.cpu().item()
