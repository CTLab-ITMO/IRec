from utils import MetaParent, create_logger
from utils import GLOBAL_TENSORBOARD_WRITER

import os
import torch
from collections import Counter

logger = create_logger(name=__name__)


class BaseCallback(metaclass=MetaParent):
    def __init__(self, model, dataloader, optimizer):
        self._model = model
        self._dataloader = dataloader
        self._optimizer = optimizer

    def __call__(self, inputs, step_num):
        raise NotImplementedError


# TODO add normal metrics, timer tensorboard, params counter

class CheckpointCallback(BaseCallback, config_name='checkpoint'):
    def __init__(self, model, dataloader, optimizer, on_step, save_path, model_name):
        super().__init__(model, dataloader, optimizer)
        self._on_step = on_step
        self._save_path = save_path  # TODO create dir if not exist
        self._model_name = model_name  # TODO create model_name dir

    @classmethod
    def create_from_config(cls, config, model=None, dataloader=None, optimizer=None):
        assert model is not None, 'Model instance should be provided'
        return cls(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            save_path=config['save_path'],
            model_name=config['model_name'],
            on_step=config['on_step']
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            torch.save(
                {
                    'step_num': step_num,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                },
                os.path.join(self._save_path, f'{self._model_name}_{step_num}.pt')  # TODO fix path
            )


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

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            running_params = Counter()
            self._model.eval()
            with torch.no_grad():
                for inputs in self._dataloader[self._dataloader_name]:
                    for key, values in inputs.items():
                        inputs[key] = torch.squeeze(inputs[key]).to('cpu')  # TODO fix
                    result = self._model(inputs)
                    for param in self._params:
                        running_params[param] += result[param]

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

    def __call__(self, inputs, step_num):
        for callback in self._callbacks:
            callback(inputs, step_num)
