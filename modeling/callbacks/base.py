from utils import MetaParent
from utils import GLOBAL_TENSORBOARD_WRITER

import torch
from collections import Counter


class BaseCallback(metaclass=MetaParent):
    def __init__(self, model, dataloader, optimizer):
        self._model = model
        self._dataloader = dataloader
        self._optimizer = optimizer

    def __call__(self, step_num):
        raise NotImplementedError


class QualityCheckCallbackCheck(BaseCallback, config_name='quality'):
    def __init__(
            self,
            model,
            dataloader,
            optimizer,
            dataloader_name,
            params,
            on_step
    ):
        super().__init__(model, dataloader, optimizer)
        self._dataloader_name = dataloader_name
        self._on_step = on_step
        self._params = params

    @classmethod
    def create_from_config(cls, config, model=None, dataloader=None, optimizer=None):
        assert model is not None, 'Model instance should be provided'
        return cls(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            dataloader_name=config['dataloader_name'],
            params=config['params'],
            on_step=config['on_step']
        )

    def __call__(self, step_num):
        running_params = Counter()

        self._model.eval()
        with torch.no_grad():
            for inputs in self._dataloader[self._dataloader_name]:
                for key, values in inputs.items():
                    inputs[key] = torch.squeeze(inputs[key]).to('cuda')  # TODO fix
                result = self._model(inputs)
                for param in self._params:
                    running_params += result[param]

        for param in self._params:
            running_params[param] /= len(self._dataloader[self._dataloader_name])
            GLOBAL_TENSORBOARD_WRITER.add_scalar(
                f'{self._dataloader_name}/{param}',
                result[param],
                step_num
            )


class CompositeCallback(BaseCallback, config_name='composite'):
    def __init__(
            self,
            model,
            dataloader,
            optimizer,
            callbacks
    ):
        super().__init__(model, dataloader, optimizer)
        self._callbacks = callbacks

    @classmethod
    def create_from_config(cls, config, model=None, dataloader=None, optimizer=None):
        return cls(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            callbacks=[
                BaseCallback.create_from_config(
                    cfg,
                    model=model,
                    dataloader=dataloader,
                    optimizer=optimizer
                ) for cfg in config['callbacks']
            ]
        )

    def __call__(self, step_num):
        for callback in self._callbacks:
            callback(step_num)
