class BaseDataset:
    def __getitem__(self, idx):
        raise NotImplemented

    def __len__(self):
        raise NotImplemented

    def map(self, mapper):
        return MapDataset(
            dataset=self,
            mapper=mapper
        )


class MapDataset(BaseDataset):
    def __init__(self, dataset, mapper):
        self.dataset = dataset
        self.mapper = mapper

    def __getitem__(self, idx):
        return self.mapper(
            self.dataset[idx]
        )

    def __len__(self):
        return len(self.dataset)