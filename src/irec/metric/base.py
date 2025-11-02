from irec.utils import MetaParent, create_masked_tensor, DEVICE

import torch


class BaseMetric(metaclass=MetaParent):
    pass


class StatefullMetric(BaseMetric):
    def reduce(self):
        raise NotImplementedError


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
        return cls(
            metrics=[
                BaseMetric.create_from_config(cfg) for cfg in config['metrics']
            ],
        )

    def __call__(self, inputs):
        for metric in self._metrics:
            inputs = metric(inputs)
        return inputs


class NDCGMetric(BaseMetric, config_name='ndcg'):
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][
            :,
            : self._k,
        ].float()  # (batch_size, top_k_indices)
        labels = inputs['{}.ids'.format(labels_prefix)].float()  # (batch_size)

        
        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(
            predictions,
            labels[..., None],
        ).float()  # (batch_size, top_k_indices)
        discount_factor = 1 / torch.log2(
            torch.arange(1, self._k + 1, 1).float() + 1.0,
        ).to(hits.device)  # (k)
        dcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size)

        return dcg.cpu().tolist()


class RecallMetric(BaseMetric, config_name='recall'):
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][
            :,
            : self._k,
        ].float()  # (batch_size, top_k_indices)
        labels = inputs['{}.ids'.format(labels_prefix)].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(
            predictions,
            labels[..., None],
        ).float()  # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()


class CoverageMetric(StatefullMetric, config_name='coverage'):
    def __init__(self, k, num_items):
        self._k = k
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(k=config['k'], num_items=kwargs['num_items'])

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][
            :,
            : self._k,
        ].float()  # (batch_size, top_k_indices)
        return (
            predictions.view(-1).long().cpu().detach().tolist()
        )  # (batch_size * k)

    def reduce(self, values):
        return len(set(values)) / self._num_items

class MCLSRNDCGMetric(BaseMetric, config_name='mclsr-ndcg'):
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].to(DEVICE) # (batch_size, k)
        labels_flat = inputs[f'{labels_prefix}.ids'].to(DEVICE)      # (total_labels,)
        labels_lengths = inputs[f'{labels_prefix}.length'].to(DEVICE) # (batch_size,)

        assert predictions.shape[0] == labels_lengths.shape[0]

        batch_size = predictions.shape[0]

        padded_labels, labels_mask = create_masked_tensor(data=labels_flat, lengths=labels_lengths)
        padded_labels[~labels_mask] = -1

        positions = torch.arange(2, self._k + 2, device=predictions.device)
        weights = 1. / torch.log2(positions.float())

        is_hit = (predictions[:, :, None] == padded_labels[:, None, :]).sum(dim=-1)  # (batch_size, k)

        num_ideal_hits = torch.minimum(labels_lengths, torch.as_tensor(self._k, device=labels_lengths.device, dtype=labels_lengths.dtype))  # (batch_size)
        ideal_mask = (torch.arange(self._k, device=is_hit.device, dtype=weights.dtype)[None, :].tile(dims=[batch_size, 1]) < num_ideal_hits[:, None])  # (batch_size, k)
        
        dcg = (is_hit.float() * weights).sum(dim=-1)  # (batch_size)
        idcg = (ideal_mask.float() * weights).sum(dim=-1)  # (batch_size)

        ndcg = dcg / idcg

        return ndcg.tolist()


class MCLSRRecallMetric(BaseMetric, config_name='mclsr-recall'):
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].to(DEVICE) # (batch_size, k)
        labels_flat = inputs[f'{labels_prefix}.ids'].to(DEVICE)      # (total_labels,)
        labels_lengths = inputs[f'{labels_prefix}.length'].to(DEVICE) # (batch_size,)

        assert predictions.shape[0] == labels_lengths.shape[0]

        padded_labels, labels_mask = create_masked_tensor(data=labels_flat, lengths=labels_lengths)
        padded_labels[~labels_mask] = -1

        is_hit = (predictions[:, :, None] == padded_labels[:, None, :]).sum(dim=-1).float()  # (batch_size, k)
        recall = is_hit.sum(dim=-1) / torch.minimum(labels_lengths, torch.as_tensor(self._k, device=labels_lengths.device, dtype=labels_lengths.dtype))  # (batch_size)
        
        return recall.tolist()

class MCLSRHitRateMetric(BaseMetric, config_name='mclsr-hit'):
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].to(DEVICE) # (batch_size, k)
        labels_flat = inputs[f'{labels_prefix}.ids'].to(DEVICE)      # (total_labels,)
        labels_lengths = inputs[f'{labels_prefix}.length'].to(DEVICE) # (batch_size,)

        assert predictions.shape[0] == labels_lengths.shape[0]

        padded_labels, labels_mask = create_masked_tensor(data=labels_flat, lengths=labels_lengths)
        padded_labels[~labels_mask] = -1
        
        hit_rate = (predictions[:, :, None] == padded_labels[:, None, :]).sum(dim=-1).max(dim=-1).values.float()  # (batch_size)
        
        return hit_rate.tolist()