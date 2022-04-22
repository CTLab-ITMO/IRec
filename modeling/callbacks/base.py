from utils import MetaParent, create_logger, maybe_to_list
from utils import GLOBAL_TENSORBOARD_WRITER, DEVICE
from utils.metrics import METRICS

import os
import torch
from pathlib import Path
from collections import Counter

logger = create_logger(name=__name__)


class BaseCallback(metaclass=MetaParent):

    def __init__(self, model, dataloader, optimizer):
        self._model = model
        self._dataloader = dataloader
        self._optimizer = optimizer

    def __call__(self, inputs, step_num):
        raise NotImplementedError


class MetricCallback(BaseCallback, config_name='metric'):

    def __init__(
            self,
            model,
            dataloader,
            optimizer,
            on_step,
            metrics
    ):
        super().__init__(model, dataloader, optimizer)
        self._on_step = on_step
        self._metrics = maybe_to_list(metrics)

    @classmethod
    def create_from_config(cls, config, model=None, dataloader=None, optimizer=None):
        assert model is not None, 'Model instance should be provided'
        return cls(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            on_step=config['on_step'],
            metrics=config['metrics']
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            for metric in self._metrics:
                metric_value = METRICS[metric](
                    ground_truth=inputs[self._model.schema['ground_truth_prefix']],
                    predictions=inputs[self._model.schema['predictions_prefix']]
                )
                GLOBAL_TENSORBOARD_WRITER.add_scalar(
                    '{}/train'.format(metric),
                    metric_value,
                    step_num
                )
            GLOBAL_TENSORBOARD_WRITER.add_scalar(
                '{}/train'.format(self._model.schema['loss_prefix']),
                inputs[self._model.schema['loss_prefix']],
                step_num
            )


class CheckpointCallback(BaseCallback, config_name='checkpoint'):

    def __init__(self, model, dataloader, optimizer, on_step, save_path, model_name):
        super().__init__(model, dataloader, optimizer)
        self._on_step = on_step
        self._save_path = Path(os.path.join(save_path, model_name))
        if self._save_path.exists():
            logger.warning('Checkpoint path `{}` is already exists!'.format(self._save_path))
        else:
            self._save_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_from_config(cls, config, model=None, dataloader=None, optimizer=None):
        assert model is not None, 'Model instance should be provided'
        return cls(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            on_step=config['on_step'],
            save_path=config['save_path'],
            model_name=config['model_name']
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            logger.debug('Saving model state on step {}...'.format(step_num))
            torch.save(
                {
                    'step_num': step_num,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                },
                os.path.join(self._save_path, 'checkpoint_{}.pt'.format(step_num))
            )
            logger.debug('Saving done!')


class QualityCheckCallbackCheck(BaseCallback, config_name='validation'):
    def __init__(
            self,
            model,
            dataloader,
            optimizer,
            on_step,
            dataloader_name,
            metrics=None
    ):
        super().__init__(model, dataloader, optimizer)
        self._on_step = on_step
        self._dataloader_name = dataloader_name
        self._metrics = maybe_to_list(metrics) if metrics is not None else []

    @classmethod
    def create_from_config(cls, config, model=None, dataloader=None, optimizer=None):
        assert model is not None, 'Model instance should be provided'
        return cls(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            on_step=config['on_step'],
            dataloader_name=config['dataloader_name'],
            metrics=config.get('metrics', None)
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            logger.debug('Validation on step {}...'.format(step_num))
            running_params = Counter()

            self._model.eval()
            with torch.no_grad():
                for inputs in self._dataloader[self._dataloader_name]:
                    for key, values in inputs.items():
                        inputs[key] = torch.squeeze(inputs[key]).to(DEVICE)
                    result = self._model(inputs)

                    for key, values in result.items():
                        result[key] = values.cpu()

                    for metric in self._metrics:
                        running_params[metric] += METRICS[metric](
                            ground_truth=inputs[self._model.schema['ground_truth_prefix']],
                            predictions=inputs[self._model.schema['predictions_prefix']]
                        )
                    running_params[self._model.schema['loss_prefix']] += inputs[
                        self._model.schema['loss_prefix']].item()

            for label, value in running_params.items():
                GLOBAL_TENSORBOARD_WRITER.add_scalar(
                    '{}/{}'.format(label, self._dataloader_name),
                    value / len(self._dataloader[self._dataloader_name]),
                    step_num
                )

            logger.debug('Validation done!')


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
