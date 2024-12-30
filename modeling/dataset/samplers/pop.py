from dataset.samplers.base import TrainSampler, EvalSampler

import copy


class PopTrainSampler(TrainSampler, config_name='pop'):
    pass


class PopEvalSampler(EvalSampler, config_name='pop'):

    def __init__(self, dataset, num_users, num_items):
        super().__init__(dataset, num_users, num_items)

        self._item_2_count = {}
        for i in range(1, num_items + 1):
            self._item_2_count[i] = 0

        for sample in dataset:
            # items = sample['item.ids'][:-1]
            items = sample['item.ids']
            for item in items:
                self._item_2_count[item] += 1

        self._candidates_counts = [0] + [
            self._item_2_count[item_id] for item_id in range(1, self._num_items + 1)
        ] + [0]

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])
        labels = [sample['item.ids'][-1]]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'labels.ids': labels,
            'labels.length': len(labels),

            'candidates_counts.ids': self._candidates_counts,
            'candidates_counts.length': len(self._candidates_counts)
        }
