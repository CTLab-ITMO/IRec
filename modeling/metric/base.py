from utils import MetaParent

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
        predictions = inputs[pred_prefix][:, :self._k].float()  # (batch_size, top_k_indices)
        labels = inputs['{}.ids'.format(labels_prefix)].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(predictions, labels[..., None]).float()  # (batch_size, top_k_indices)
        discount_factor = 1 / torch.log2(torch.arange(1, self._k + 1, 1).float() + 1.).to(hits.device)  # (k)
        dcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size)

        return dcg.cpu().tolist()
    

class NDCGSemanticMetric(BaseMetric, config_name='ndcg_semantic'):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].float()  # (batch_size, top_k_indices)
        labels = inputs['{}.ids'.format(labels_prefix)].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(predictions, labels[..., None]).float()  # (batch_size, top_k_indices)
        discount_factor = 1 / torch.log2(torch.arange(1, self._k + 1, 1).float() + 1.).to(hits.device)  # (k)
        dcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size)

        return dcg.cpu().tolist()


class RecallMetric(BaseMetric, config_name='recall'):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].float()  # (batch_size, top_k_indices)
        labels = inputs['{}.ids'.format(labels_prefix)].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(predictions, labels[..., None]).float()  # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()


class RecallSemanticMetric(BaseMetric, config_name='recall_semantic'):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        import code; code.interact(local=locals())
        labels = inputs['{}.ids'.format(labels_prefix)].float()  # (batch_size)

        all_items_semantic_ids = inputs['all_semantic_ids']  # (num_items, sid_length)
        all_items_semantic_ids = all_items_semantic_ids + 256 * torch.arange(4)

        decoder_scores = torch.stack([inputs[f'decoder_scores_{i}'] for i in range(0 + 1, 4 + 1)], dim=1)
        decoder_scores = decoder_scores.reshape(decoder_scores.shape[0], decoder_scores.shape[1] * decoder_scores.shape[2])

        batch_size = labels.shape[0]
        all_items, id_dim = all_items_semantic_ids.shape
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
        ids_expanded = all_items_semantic_ids.unsqueeze(0).expand(batch_size, -1, -1)

        all_item_scores = decoder_scores[batch_indices.expand(-1, all_items, id_dim), ids_expanded]  # (batch_size, num_items, sid_length)

        batch_size, num_items, sid_length = all_item_scores.shape
    
        indices = torch.arange(num_items).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 4)
        
        for i in range(sid_length - 1, -1, -1):  # sid_length-1, sid_length-2, ..., 1, 0
            key_values = torch.gather(all_item_scores, dim=1, index=indices)[:, :, i]
            sort_indices = torch.argsort(key_values, dim=1, descending=True, stable=True).unsqueeze(-1).expand(batch_size, -1, 4)
            indices = torch.gather(indices, dim=1, index=sort_indices)
        
        print(indices[:2, :2])
        indices = indices[:, :, 0]

        predictions = indices[:, :self._k].float()  # (batch_size, top_k_indices)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(predictions, labels[..., None]).float()  # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()


class CoverageMetric(StatefullMetric, config_name='coverage'):

    def __init__(self, k, num_items):
        self._k = k
        self._num_items = num_items
    
    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            k=config['k'],
            num_items=kwargs['num_items']
        )

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].float()  # (batch_size, top_k_indices)
        return predictions.view(-1).long().cpu().detach().tolist()  # (batch_size * k)
    
    def reduce(self, values):
        return len(set(values)) / self._num_items
