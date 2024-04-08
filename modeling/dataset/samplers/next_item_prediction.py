from dataset.samplers.base import TrainSampler, ValidationSampler, EvalSampler
from dataset.samplers.base import MultiDomainTrainSampler, MultiDomainValidationSampler, MultiDomainEvalSampler, \
    NegativeRatingsTrainSampler, NegativeRatingsValidationSampler, NegativeRatingsEvalSampler
from dataset.negative_samplers import BaseNegativeSampler, BaseNegRatingsNegativeSampler

import copy


class NextItemPredictionTrainSampler(TrainSampler, config_name='next_item_prediction'):

    def __init__(self, dataset, num_users, num_items, negative_sampler, num_negatives=-1):
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
            num_negatives=config.get('num_negatives_train', -1)
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]
        next_item_sequence = sample['item.ids'][1:]

        if self._num_negatives == -1:
            negative_sequence = self._negative_sampler.generate_negative_samples(
                sample, len(next_item_sequence)
            )
        else:
            negative_sequence = self._negative_sampler.generate_negative_samples(
                sample, self._num_negatives
            )

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

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'candidates.ids': candidates,
            'candidates.length': len(candidates),

            'labels.ids': [0],
            'labels.length': 1,
        }


class NextItemPredictionEvalSampler(EvalSampler, config_name='next_item_prediction'):

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )


class MultiDomainNextItemPredictionTrainSampler(MultiDomainTrainSampler,
                                                config_name='multi_domain_next_item_prediction'):

    def __init__(
            self,
            dataset,
            num_users,
            num_items,
            target_domain,
            other_domains,
            negative_sampler
    ):

        super().__init__(target_domain, other_domains)
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._user_id_to_index_cross_domain_mapping = self.get_user_id_to_index_cross_domain_mapping()

    def get_user_id_to_index_cross_domain_mapping(self):
        _user_id_to_index_cross_domain_mapping = {domain: {} for domain in self._other_domains}
        for domain in self._other_domains:
            for index, sample in enumerate(self._dataset[domain]):
                user_id = sample['user.ids'][0]
                _user_id_to_index_cross_domain_mapping[domain][index] = user_id

        return _user_id_to_index_cross_domain_mapping

    @classmethod
    def create_from_config(cls, config, **kwargs):
        domains = [config['target_domain']] + config['other_domains']
        negative_sampler = {}

        datasets = kwargs['dataset']
        for domain in domains:
            kwargs['dataset'] = datasets[domain]
            negative_sampler[domain] = BaseNegativeSampler.create_from_config(
                {'type': config['negative_sampler_type']},
                **kwargs
            )
        kwargs['dataset'] = datasets

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler,
            target_domain=config['target_domain'],
            other_domains=config['other_domains']
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
            'negative.length': len(negative_sequence)
        }

        # other domains
        for domain in self._other_domains:
            domain_user_index = self._user_id_to_index_cross_domain_mapping[domain][index]
            sample = copy.deepcopy(self._dataset[domain][domain_user_index])

            item_sequence = sample['item.ids']
            next_item_sequence = sample['item.ids'][1:]
            negative_sequence = self._negative_sampler[domain].generate_negative_samples(sample,
                                                                                         len(next_item_sequence))

            assert len(next_item_sequence) == len(negative_sequence)

            result.update({
                'item.{}.ids'.format(domain): item_sequence,
                'item.{}.length'.format(domain): len(item_sequence),

                'positive.{}.ids'.format(domain): next_item_sequence,
                'positive.{}.length'.format(domain): len(next_item_sequence),

                'negative.{}.ids'.format(domain): negative_sequence,
                'negative.{}.length'.format(domain): len(negative_sequence)
            })

        return result


class MultiDomainNextItemPredictionValidationSampler(MultiDomainValidationSampler,
                                                     config_name='multi_domain_next_item_prediction'):

    def __init__(
            self,
            dataset,
            num_users,
            num_items,
            target_domain,
            other_domains,
            negative_sampler,
            num_negatives=100
    ):

        super().__init__(target_domain, other_domains)
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._num_negatives = num_negatives
        self._user_id_to_index_cross_domain_mapping = self.get_user_id_to_index_cross_domain_mapping()

    def get_user_id_to_index_cross_domain_mapping(self):
        _user_id_to_index_cross_domain_mapping = {domain: {} for domain in self._other_domains}
        for domain in self._other_domains:
            for index, sample in enumerate(self._dataset[domain]):
                user_id = sample['user.ids'][0]
                _user_id_to_index_cross_domain_mapping[domain][index] = user_id

        return _user_id_to_index_cross_domain_mapping

    @classmethod
    def create_from_config(cls, config, **kwargs):
        domains = [config['target_domain']] + config['other_domains']
        negative_sampler = {}

        datasets = kwargs['dataset']
        for domain in domains:
            kwargs['dataset'] = datasets[domain]
            negative_sampler[domain] = BaseNegativeSampler.create_from_config(
                {'type': config['negative_sampler_type']},
                **kwargs
            )
        kwargs['dataset'] = datasets

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler,
            num_negatives=config.get('num_negatives_val', 100),
            target_domain=config['target_domain'],
            other_domains=config['other_domains']
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
            'labels.length': len(labels)
        }

        # other domains
        for domain in self._other_domains:
            domain_user_index = self._user_id_to_index_cross_domain_mapping[domain][index]
            sample = copy.deepcopy(self._dataset[domain][domain_user_index])

            item_sequence = sample['item.ids'][:-1]

            positive = sample['item.ids'][-1]
            negatives = self._negative_sampler[domain].generate_negative_samples(sample, self._num_negatives)

            candidates = [positive] + negatives
            labels = [1] + [0] * len(negatives)

            result.update({
                'item.{}.ids'.format(domain): item_sequence,
                'item.{}.length'.format(domain): len(item_sequence),

                'candidates.{}.ids'.format(domain): candidates,
                'candidates.{}.length'.format(domain): len(candidates),

                'labels.{}.ids'.format(domain): labels,
                'labels.{}.length'.format(domain): len(labels)
            })

        return result


class MultiDomainNextItemPredictionEvalSampler(MultiDomainEvalSampler, config_name='multi_domain_next_item_prediction'):

    def __init__(
            self,
            dataset,
            num_users,
            num_items,
            target_domain,
            other_domains
    ):

        super().__init__(dataset, num_users, num_items, target_domain, other_domains)
        self._user_id_to_index_cross_domain_mapping = self.get_user_id_to_index_cross_domain_mapping()

    def get_user_id_to_index_cross_domain_mapping(self):
        _user_id_to_index_cross_domain_mapping = {domain: {} for domain in self._other_domains}
        for domain in self._other_domains:
            for index, sample in enumerate(self._dataset[domain]):
                user_id = sample['user.ids'][0]
                _user_id_to_index_cross_domain_mapping[domain][index] = user_id

        return _user_id_to_index_cross_domain_mapping

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            target_domain=config['target_domain'],
            other_domains=config['other_domains']
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
            'labels.length': 1
        }

        # other domains
        for domain in self._other_domains:
            domain_user_index = self._user_id_to_index_cross_domain_mapping[domain][index]
            sample = copy.deepcopy(self._dataset[domain][domain_user_index])

            item_sequence = sample['item.ids'][:-1]
            next_item = sample['item.ids'][-1]

            result.update({
                'item.{}.ids'.format(domain): item_sequence,
                'item.{}.length'.format(domain): len(item_sequence),

                'labels.{}.ids'.format(domain): [next_item],
                'labels.{}.length'.format(domain): 1
            })

        return result


class NegativeRatingsNextItemPredictionTrainSampler(NegativeRatingsTrainSampler,
                                                    config_name='negative_ratings_next_item_prediction'):

    def __init__(
            self,
            dataset,
            num_users,
            num_items,
            positive_domain,
            negative_domain,
            num_negatives,
            negative_sampler,
            offset
    ):

        super().__init__(positive_domain, negative_domain)
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._num_negatives = num_negatives
        self._negative_sampler = negative_sampler
        self._offset = offset
        self._user_id_to_index_cross_domain_mapping = self.get_user_id_to_index_cross_domain_mapping()

    def get_user_id_to_index_cross_domain_mapping(self):
        _user_id_to_index_cross_domain_mapping = {self._negative_domain: {}}
        for index, sample in enumerate(self._dataset[self._negative_domain]):
            user_id = sample['user.ids'][0]
            _user_id_to_index_cross_domain_mapping[self._negative_domain][user_id] = index

        return _user_id_to_index_cross_domain_mapping

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegRatingsNegativeSampler.create_from_config({'type': config['negative_sampler_type']}, **kwargs)

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            positive_domain=config['positive_domain'],
            negative_domain=config['negative_domain'],
            negative_sampler=negative_sampler,
            num_negatives=config.get('num_negatives_train', -1),
            offset=config['offset']
        )

    def __getitem__(self, index):
        sample_positive_domain = copy.deepcopy(self._dataset[self._positive_domain][index])
        item_sequence_positive_domain = sample_positive_domain['item.ids'][:-1]
        ratings_positive_domain = sample_positive_domain['ratings.ids'][1:]
        next_item_sequence = sample_positive_domain['item.ids'][1:]

        item_sequence_negative_domain = []
        ratings_negative_domain = []
        if sample_positive_domain['user.ids'][0] in self._user_id_to_index_cross_domain_mapping[self._negative_domain]:
            domain_index_negative_domain = self._user_id_to_index_cross_domain_mapping[self._negative_domain][sample_positive_domain['user.ids'][0]]
            sample_negative_domain = copy.deepcopy(self._dataset[self._negative_domain][domain_index_negative_domain])
            item_sequence_negative_domain = sample_negative_domain['item.ids'][:-1]
            ratings_negative_domain = sample_negative_domain['ratings.ids'][:-1]

        item_sequence = item_sequence_positive_domain + item_sequence_negative_domain
        ratings = ratings_positive_domain + ratings_negative_domain
        ratings = [rating - self._offset for rating in ratings]

        len_item_sequence = len(item_sequence)

        if self._num_negatives == -1:
            negatives = self._negative_sampler.generate_negative_samples(sample_positive_domain, len_item_sequence)
        else:
            negatives = self._negative_sampler.generate_negative_samples(sample_positive_domain, self._num_negatives)

        # TODO размерность batch_size * num_negatives должно быть ровно = negatives.ids
        result = {
            'user.ids': sample_positive_domain['user.ids'] * len_item_sequence,
            'user.length': len_item_sequence,

            'item.ids': item_sequence,
            'item.length': len_item_sequence,

            'ratings.ids': ratings,
            'ratings.length': len(ratings),

            'negatives.ids': negatives,
            'negatives.length': len(negatives),

            'user.graph.ids': sample_positive_domain['user.ids'],
            'user.graph.length': len(sample_positive_domain['user.ids']),

            'item.graph.ids': item_sequence_positive_domain,
            'item.graph.length': len(item_sequence_positive_domain)
        }

        return result


class NegativeRatingsNextItemPredictionValidationSampler(NegativeRatingsValidationSampler,
                                                         config_name='negative_ratings_next_item_prediction'):
    def __init__(
            self,
            dataset,
            num_users,
            num_items,
            positive_domain,
            negative_domain,
            negative_sampler,
            num_negatives,
            offset
    ):

        super().__init__(positive_domain, negative_domain)
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._num_negatives = num_negatives
        self._offset = offset
        self._user_id_to_index_cross_domain_mapping = self.get_user_id_to_index_cross_domain_mapping()

    def get_user_id_to_index_cross_domain_mapping(self):
        _user_id_to_index_cross_domain_mapping = {self._negative_domain: {}}
        for index, sample in enumerate(self._dataset[self._negative_domain]):
            user_id = sample['user.ids'][0]
            _user_id_to_index_cross_domain_mapping[self._negative_domain][user_id] = index

        return _user_id_to_index_cross_domain_mapping

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegRatingsNegativeSampler.create_from_config({'type': config['negative_sampler_type']}, **kwargs)

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler,
            num_negatives=config.get('num_negatives_val', -1),
            positive_domain=config['positive_domain'],
            negative_domain=config['negative_domain'],
            offset=config['offset']
        )

    def __getitem__(self, index):
        sample_positive_domain = copy.deepcopy(self._dataset[self._positive_domain][index])
        item_sequence_positive_domain = sample_positive_domain['item.ids'][:-1]
        ratings_positive_domain = sample_positive_domain['ratings.ids'][:-1]
        positive_graph = sample_positive_domain['item.ids'][-1]

        item_sequence_negative_domain = []
        ratings_negative_domain = []
        if sample_positive_domain['user.ids'][0] in self._user_id_to_index_cross_domain_mapping[self._negative_domain]:
            domain_index_negative_domain = self._user_id_to_index_cross_domain_mapping[self._negative_domain][sample_positive_domain['user.ids'][0]]
            sample_negative_domain = copy.deepcopy(self._dataset[self._negative_domain][domain_index_negative_domain])
            item_sequence_negative_domain = sample_negative_domain['item.ids'][:-1]
            ratings_negative_domain = sample_negative_domain['ratings.ids'][:-1]

        item_sequence = item_sequence_positive_domain + item_sequence_negative_domain
        ratings = ratings_positive_domain + ratings_negative_domain
        ratings = [rating - self._offset for rating in ratings]

        len_item_sequence = len(item_sequence)

        if self._num_negatives == -1:
            negatives = self._negative_sampler.generate_negative_samples(sample_positive_domain, len_item_sequence)
        else:
            negatives = self._negative_sampler.generate_negative_samples(sample_positive_domain, self._num_negatives)

        candidates_graph = [positive_graph] + negatives.tolist()

        result = {
            'user.ids': sample_positive_domain['user.ids'] * len_item_sequence,
            'user.length': len_item_sequence,

            'item.ids': item_sequence,
            'item.length': len_item_sequence,

            'ratings.ids': ratings,
            'ratings.length': len(ratings),

            'user.graph.ids': sample_positive_domain['user.ids'],
            'user.graph.length': len(sample_positive_domain['user.ids']),

            'item.graph.ids': item_sequence_positive_domain,
            'item.graph.length': len(item_sequence_positive_domain),

            'candidates.graph.ids': candidates_graph,
            'candidates.graph.length': len(candidates_graph),

            'labels.graph.ids': [0],
            'labels.graph.length': 1
        }

        return result


class NegativeRatingsNextItemPredictionEvalSampler(NegativeRatingsEvalSampler,
                                                   config_name='negative_ratings_next_item_prediction'):
    def __init__(
            self,
            dataset,
            num_users,
            num_items,
            positive_domain,
            negative_domain,
            offset
    ):

        super().__init__(dataset, num_users, num_items, positive_domain, negative_domain)
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._offset = offset
        self._user_id_to_index_cross_domain_mapping = self.get_user_id_to_index_cross_domain_mapping()

    def get_user_id_to_index_cross_domain_mapping(self):
        _user_id_to_index_cross_domain_mapping = {self._negative_domain: {}}
        for index, sample in enumerate(self._dataset[self._negative_domain]):
            user_id = sample['user.ids'][0]
            _user_id_to_index_cross_domain_mapping[self._negative_domain][user_id] = index

        return _user_id_to_index_cross_domain_mapping

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            positive_domain=config['positive_domain'],
            negative_domain=config['negative_domain'],
            offset=config['offset']
        )

    def __getitem__(self, index):
        sample_positive_domain = copy.deepcopy(self._dataset[self._positive_domain][index])
        item_sequence_positive_domain = sample_positive_domain['item.ids'][:-1]
        ratings_positive_domain = sample_positive_domain['ratings.ids'][:-1]

        next_item_graph = sample_positive_domain['item.ids'][-1]

        item_sequence_negative_domain = []
        ratings_negative_domain = []
        if sample_positive_domain['user.ids'][0] in self._user_id_to_index_cross_domain_mapping[self._negative_domain]:
            domain_index_negative_domain = self._user_id_to_index_cross_domain_mapping[self._negative_domain][sample_positive_domain['user.ids'][0]]
            sample_negative_domain = copy.deepcopy(self._dataset[self._negative_domain][domain_index_negative_domain])
            item_sequence_negative_domain = sample_negative_domain['item.ids'][:-1]
            ratings_negative_domain = sample_negative_domain['ratings.ids'][:-1]

        item_sequence = item_sequence_positive_domain + item_sequence_negative_domain
        ratings = ratings_positive_domain + ratings_negative_domain
        ratings = [rating - self._offset for rating in ratings]

        len_item_sequence = len(item_sequence)
        result = {
            'user.ids': sample_positive_domain['user.ids'] * len_item_sequence,
            'user.length': len_item_sequence,

            'item.ids': item_sequence,
            'item.length': len_item_sequence,

            'ratings.ids': ratings,
            'ratings.length': len(ratings),

            'user.graph.ids': sample_positive_domain['user.ids'],
            'user.graph.length': len(sample_positive_domain['user.ids']),

            'item.graph.ids': item_sequence_positive_domain,
            'item.graph.length': len(item_sequence_positive_domain),

            'labels.graph.ids': [next_item_graph],
            'labels.graph.length': 1
        }

        return result
