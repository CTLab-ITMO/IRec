import torch

from utils import MetaParent


class BaseMetric(metaclass=MetaParent):
    pass


class StatefullMetric(BaseMetric):
    def reduce(self):
        raise NotImplementedError


class StaticMetric(BaseMetric, config_name="dummy"):
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def __call__(self, inputs):
        inputs[self._name] = self._value

        return inputs


class CompositeMetric(BaseMetric, config_name="composite"):
    def __init__(self, metrics):
        self._metrics = metrics

    @classmethod
    def create_from_config(cls, config):
        return cls(
            metrics=[BaseMetric.create_from_config(cfg) for cfg in config["metrics"]]
        )

    def __call__(self, inputs):
        for metric in self._metrics:
            inputs = metric(inputs)
        return inputs


class NDCGMetric(BaseMetric, config_name="ndcg"):
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][
            :, : self._k
        ].float()  # (batch_size, top_k_indices)
        labels = inputs["{}.ids".format(labels_prefix)].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(
            predictions, labels[..., None]
        ).float()  # (batch_size, top_k_indices)
        discount_factor = 1 / torch.log2(
            torch.arange(1, self._k + 1, 1).float() + 1.0
        ).to(hits.device)  # (k)
        # print(f"hits {sum(hits)}, discount factor {discount_factor}")
        dcg = torch.einsum("bk,k->b", hits, discount_factor)  # (batch_size)

        return dcg.cpu().tolist()


class RecallMetric(BaseMetric, config_name="recall"):
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][
            :, : self._k
        ].float()  # (batch_size, top_k_indices)
        labels = inputs["{}.ids".format(labels_prefix)].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(
            predictions, labels[..., None]
        ).float()  # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()


class CoverageMetric(StatefullMetric, config_name="coverage"):
    def __init__(self, k, num_items):
        self._k = k
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(k=config["k"], num_items=kwargs["num_items"])

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][
            :, : self._k
        ].float()  # (batch_size, top_k_indices)
        return predictions.view(-1).long().cpu().detach().tolist()  # (batch_size * k)

    def reduce(self, values):
        return len(set(values)) / self._num_items
