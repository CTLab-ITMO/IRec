from dataset.samplers.base import TrainSampler, ValidationSampler, EvalSampler
from dataset.negative_samplers.base import BaseNegativeSampler

import copy


class NextItemPredictionTrainSampler(TrainSampler, config_name='next_item_prediction'):

    def __init__(self, dataset, num_users, num_items, negative_sampler):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config({'type': config['negative_sampler_type']}, **kwargs)

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]
        next_item_sequence = sample['item.ids'][1:]
        negative_sequence = self._negative_sampler.generate_negative_samples(sample, len(next_item_sequence))

        assert len(next_item_sequence) == len(negative_sequence)

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'positive.ids': next_item_sequence,
            'positive.length': len(next_item_sequence),

            'negative.ids': negative_sequence,
            'negative.length': len(negative_sequence)
        }


class NextItemPredictionValidationSampler(ValidationSampler, config_name='next_item_prediction'):

    def __init__(self, dataset, num_users, num_items, negative_sampler, num_negatives=100):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._num_negatives = num_negatives

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config({'type': config['negative_sampler_type']}, **kwargs)

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler,
            num_negatives=config.get('num_negatives_val', 100)
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]

        positive = sample['item.ids'][-1]
        negatives = self._negative_sampler.generate_negative_samples(sample, self._num_negatives)

        candidates = [positive] + negatives
        labels = [1] + [0] * len(negatives)

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'candidates.ids': candidates,
            'candidates.length': len(candidates),

            'labels.ids': labels,
            'labels.length': len(labels),
        }


class NextItemPredictionEvalSampler(EvalSampler, config_name='next_item_prediction'):

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )


class MultiDomainNextItemPredictionTrainSampler(TrainSampler, config_name='multi_domain_next_item_prediction'):

    def __init__(self, dataset, num_users, num_items, target_domain, other_domains, negative_sampler):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._target_domain = target_domain
        self._other_domains = other_domains

        self._user_id_to_idx_cross_domain_mapping = {domain:{} for domain in self._other_domains}
        for domain in self._other_domains:
            for idx, el in enumerate(self._dataset[domain]):
                u_id = el['user.ids'][0]
                self._user_id_to_idx_cross_domain_mapping[domain][u_id] = idx

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config({'type': config['negative_sampler_type']}, **kwargs)

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler,
            target_domain = kwargs['target_domain'],
            other_domains = kwargs['other_domains']
        )

    def __getitem__(self, index):
        # target domain
        sample = copy.deepcopy(self._dataset[self._target_domain][index])

        item_sequence = sample['item.ids'][:-1]
        next_item_sequence = sample['item.ids'][1:]
        negative_sequence = self._negative_sampler.generate_negative_samples(sample, len(next_item_sequence))

        assert len(next_item_sequence) == len(negative_sequence)

        result = {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'positive.ids': next_item_sequence,
            'positive.length': len(next_item_sequence),

            'negative.ids': negative_sequence,
            'negative.length': len(negative_sequence),

            'other_domains.parameters': {}
        }

        # other domains
        for domain in self._other_domains:
            domain_user_index = self._user_id_to_idx_cross_domain_mapping[domain][index]
            sample = copy.deepcopy(self._dataset[domain][domain_user_index])

            item_sequence = sample['item.ids']
            next_item_sequence = sample['item.ids'][1:]
            negative_sequence = self._negative_sampler.generate_negative_samples(sample, len(next_item_sequence))

            assert len(next_item_sequence) == len(negative_sequence)

            result['other_domains.parameters'][domain] = {
                'user.ids': sample['user.ids'],
                'user.length': sample['user.length'],

                'item.ids': item_sequence,
                'item.length': len(item_sequence),

                'positive.ids': next_item_sequence,
                'positive.length': len(next_item_sequence),

                'negative.ids': negative_sequence,
                'negative.length': len(negative_sequence)
            }

        return result


class MultiDomainNextItemPredictionValidationSampler(ValidationSampler, config_name='multi_domain_next_item_prediction'):

    def __init__(self, dataset, num_users, num_items, target_domain, other_domains, negative_sampler, num_negatives=100):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._num_negatives = num_negatives
        self._target_domain = target_domain
        self._other_domains = other_domains

        self._user_id_to_idx_cross_domain_mapping = {domain:{} for domain in self._other_domains}
        for domain in self._other_domains:
            for idx, el in enumerate(self._dataset[domain]):
                u_id = el['user.ids'][0]
                self._user_id_to_idx_cross_domain_mapping[domain][u_id] = idx

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config({'type': config['negative_sampler_type']}, **kwargs)

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler,
            num_negatives=config.get('num_negatives_val', 100),
            target_domain = kwargs['target_domain'],
            other_domains = kwargs['other_domains']
        )

    def __getitem__(self, index):
        # target domain
        sample = copy.deepcopy(self._dataset[self._target_domain][index])

        item_sequence = sample['item.ids'][:-1]

        positive = sample['item.ids'][-1]
        negatives = self._negative_sampler.generate_negative_samples(sample, self._num_negatives)

        candidates = [positive] + negatives
        labels = [1] + [0] * len(negatives)

        result = {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'candidates.ids': candidates,
            'candidates.length': len(candidates),

            'labels.ids': labels,
            'labels.length': len(labels),

            'other_domains.parameters': {}
        }

        # other domains
        for domain in self._other_domains:
            domain_user_index = self._user_id_to_idx_cross_domain_mapping[domain][index]
            sample = copy.deepcopy(self._dataset[domain][domain_user_index])

            item_sequence = sample['item.ids'][:-1]

            positive = sample['item.ids'][-1]
            negatives = self._negative_sampler.generate_negative_samples(sample, self._num_negatives)

            candidates = [positive] + negatives
            labels = [1] + [0] * len(negatives)

            result['other_domains.parameters'][domain] = {
                'user.ids': sample['user.ids'],
                'user.length': sample['user.length'],

                'item.ids': item_sequence,
                'item.length': len(item_sequence),

                'candidates.ids': candidates,
                'candidates.length': len(candidates),

                'labels.ids': labels,
                'labels.length': len(labels)
                }

        return result


class MultiDomainNextItemPredictionEvalSampler(EvalSampler, config_name='multi_domain_next_item_prediction'):

    def __init__(self, dataset, num_users, num_items, target_domain, other_domains):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._target_domain = target_domain
        self._other_domains = other_domains

        self._user_id_to_idx_cross_domain_mapping = {domain:{} for domain in self._other_domains}
        for domain in self._other_domains:
            for idx, el in enumerate(self._dataset[domain]):
                u_id = el['user.ids'][0]
                self._user_id_to_idx_cross_domain_mapping[domain][u_id] = idx

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            target_domain = kwargs['target_domain'],
            other_domains = kwargs['other_domains']
        )

    def __getitem__(self, index):
        # target domain
        sample = copy.deepcopy(self._dataset[self._target_domain][index])

        item_sequence = sample['item.ids'][:-1]
        next_item = sample['item.ids'][-1]

        result = {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'labels.ids': [next_item],
            'labels.length': 1,

            'other_domains.parameters': {}
        }

        # other domains
        for domain in self._other_domains:
            domain_user_index = self._user_id_to_idx_cross_domain_mapping[domain][index]
            sample = copy.deepcopy(self._dataset[domain][domain_user_index])

            item_sequence = sample['item.ids'][:-1]
            next_item = sample['item.ids'][-1]

            result['other_domains.parameters'][domain] = {
                'user.ids': sample['user.ids'],
                'user.length': sample['user.length'],

                'item.ids': item_sequence,
                'item.length': len(item_sequence),

                'labels.ids': [next_item],
                'labels.length': 1
            }

        return result