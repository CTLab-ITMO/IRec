import copy
from loguru import logger
import os

import torch

from irec.callbacks.base import Callback
from irec.runners.base import Runner, RunnerContext


# TODO сделать для сохранения модели отдельный callback

class EarlyStopping(Callback):
    def __init__(
            self, 
            metric, 
            patience, 
            *, 
            minimize=True,
            model_path=None,
        ):
        self._metric = metric
        self._best_metric = None
        self._minimize = minimize

        self._patience = patience
        self._wait = 0

        self._best_model_state_dict = None
        self._model_path = model_path

    def state_dict(self):
        return {
            'wait': self._wait,
            'best_metric': self._best_metric
        }

    def load_state_dict(self, state_dict):
        self._wait = state_dict['wait']
        self._best_metric = state_dict['best_metric']

    def after_step(self, runner: Runner, context: RunnerContext):
        assert self._metric in context.metrics
        metric = context.metrics[self._metric]
        if self._best_metric is None:
            self._best_metric = metric
            save_path = f'{self._model_path}_best_{round(self._best_metric, 4)}.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(runner.model.state_dict(), save_path)
        else:
            if (self._minimize and metric < self._best_metric) or (not self._minimize and metric > self._best_metric):
                self._wait = 0
                old_metric = self._best_metric
                self._best_metric = metric
                # Saving new model
                save_path = f'{self._model_path}_best_{round(self._best_metric, 4)}.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(runner.model.state_dict(), save_path)
                # Deleting old model
                if str(round(self._best_metric, 4)) != str(round(old_metric, 4)):
                    os.remove(f'{self._model_path}_best_{round(old_metric, 4)}.pth')
                logger.info(f'New best value for {self._metric}: {self._best_metric:.4f}')
            else:
                self._wait += 1
                logger.info(f'Wait is increased to {self._wait}')
        if self._wait == self._patience:
            logger.info(f'Patience for {self._metric} is reached: couldn"t beat value {self._best_metric:.4f} for {self._wait} calls')
            raise StopIteration


class StopAfterNumSteps(Callback):
    def __init__(self, num_steps):
        self._num_steps = num_steps

    def after_step(self, runner: Runner, context: RunnerContext):
        if runner.global_step >= self._num_steps:
            raise StopIteration

