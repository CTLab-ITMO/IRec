from irec.utils import MetaParent

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
        predictions = inputs[pred_prefix][:, :self._k] # (batch_size, k)
        labels_flat = inputs[f'{labels_prefix}.ids']      # (total_labels,)
        labels_lengths = inputs[f'{labels_prefix}.length'] # (batch_size,)

        assert predictions.shape[0] == labels_lengths.shape[0]

        dcg_scores = []
        offset = 0
        for i in range(predictions.shape[0]):
            k = min(self._k, labels_lengths[i])

            user_predictions = predictions[i]
            num_user_labels = labels_lengths[i]
            user_labels = labels_flat[offset : offset + num_user_labels]
            offset += num_user_labels

            hits_mask = torch.isin(user_predictions, user_labels) # (k,) -> True/False
            
            positions = torch.arange(2, self._k + 2, device=predictions.device)
            weights = 1 / torch.log2(positions.float())
            dcg = (hits_mask.float() * weights).sum()
            
            idcg_weights = weights[:k]
            idcg = idcg_weights.sum()
            
            ndcg = dcg / idcg if idcg > 0 else torch.tensor(0.0)
            dcg_scores.append(ndcg.cpu().item())
            
        return dcg_scores


class MCLSRRecallMetric(BaseMetric, config_name='mclsr-recall'):
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k] # (batch_size, k)
        labels_flat = inputs[f'{labels_prefix}.ids']      # (total_labels,)
        labels_lengths = inputs[f'{labels_prefix}.length'] # (batch_size,)

        assert predictions.shape[0] == labels_lengths.shape[0]

        recall_scores = []
        offset = 0
        for i in range(predictions.shape[0]):
            user_predictions = predictions[i]
            num_user_labels = labels_lengths[i]
            user_labels = labels_flat[offset : offset + num_user_labels]
            offset += num_user_labels
            
            hits = torch.isin(user_predictions, user_labels).sum().float()
            
            recall = hits / num_user_labels if num_user_labels > 0 else torch.tensor(0.0)
            recall_scores.append(recall.cpu().item())
        
        return recall_scores