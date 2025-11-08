import numpy as np
import torch


class Transform:
    def __call__(self, sample):
        raise NotImplemented


class Mapper(Transform):
    def __call__(self, sample):
        for k, v in sample.items():
            if isinstance(v, dict):
                sample[k] = self.__call__(v)
            else:
                sample[k] = self._mapper(v)
        return sample
    
    def _mapper(self, x):
        raise NotImplemented


class ToDevice(Mapper):
    def __init__(self, device):
        self.device=device

    def _mapper(self, value):
        assert isinstance(value, torch.Tensor)
        return value.to(self.device)


class ToTorch(Mapper):
    def _mapper(self, value):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        elif isinstance(value, list):
            return torch.tensor(value)
        elif isinstance(value, torch.Tensor):
            return value
        elif isinstance(value, int) or isinstance(value, float):
            return torch.as_tensor(value)
        else:
            assert False


class Collate(Transform):
    def __call__(self, batch):
        assert batch and isinstance(batch, list), batch
        processed_batch = {}

        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            if isinstance(values[0], dict):
                processed_batch[key] = self.__call__(values)
            elif isinstance(values[0], np.ndarray):
                processed_batch[key] = np.empty(shape=(0,), dtype=values[0].dtype)
                values = [value for value in values if value.size > 0]
                if len(values) > 0:
                    processed_batch[key] = np.concatenate(values)
            elif isinstance(values[0], torch.Tensor):
                processed_batch[key] = torch.empty(size=(0,), dtype=values[0].dtype)
                values = [value for value in values if value.numel() > 0]
                if len(values) > 0:
                    if values[0].ndim == 0:  # These are numbers
                        processed_batch[key] = torch.stack(values)
                    else:
                        processed_batch[key] = torch.cat(values)
            else:
                processed_batch[key] = np.array(values)
        return processed_batch
    



