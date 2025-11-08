import math

import torch
from loguru import logger

from irec.callbacks.base import BatchCallback, Callback, LambdaCallback
from irec.callbacks.train import TrainingCallback

from irec.runners.base import Runner, RunnerContext
from irec.runners.train import TrainingRunner, TrainingRunnerContext

from irec.runners.inference import InferenceRunner


class BatchMetrics(TrainingCallback):
    def __init__(self, metrics, name=None, separator='/'):
        self._metrics = metrics
        self._name = name
        self._separator = separator

    def after_step(self, runner: TrainingRunner, context: TrainingRunnerContext):
        metrics = self._metrics(context.model_outputs, context.batch)
        BatchMetrics.add_context_metrics(context, metrics, name=self._name, separator=self._separator)
    
    @staticmethod
    def add_context_metrics(context: TrainingRunnerContext, metrics, *, name=None, separator='/'):
        metrics = {name: metrics} if name is not None else metrics
        metrics = BatchMetrics.flatten_nested_metrics(metrics, separator=separator)

        for metric_name, metric_value in metrics.items():
            if metric_name in context.metrics:
                raise ValueError(f'Metric already exists: {metric_name}')

            assert isinstance(metric_value, list) or isinstance(metric_value, float) or isinstance(metric_value, int)

            context.metrics[metric_name] = metric_value

    @staticmethod
    def flatten_nested_metrics(metrics, *, separator='/'):
        if not isinstance(metrics, dict):
            raise TypeError('If name is None metrics must return dict')
        result = dict()
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                for key, value in BatchMetrics.flatten_nested_metrics(metric_value, separator=separator).items():
                    result[metric_name + separator + key] = value
            else:
                result[metric_name] = metric_value
        return result


class LambdaMetrics(LambdaCallback):
    def __init__(self, function, name=None, separator='/'):
        super().__init__(function)
        self._name = name
        self._separator = separator

    def after_step(self, runner: Runner, context: RunnerContext):
        metrics = self._emit(runner, context)
        BatchMetrics.add_context_metrics(context, metrics, name=self._name, separator=self._separator)


class Accumulator:
    def accumulate(self, value):
        pass
    
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass

    def reduce(self):
        pass

    def clear(self):
        pass


class MeanAccumulator(Accumulator):
    def __init__(self):
        super().__init__()
        self._accumulated_values = []
    
    def accumulate(self, values):
        if isinstance(values, list):
            self._accumulated_values.extend(values)
        else:
            self._accumulated_values.append(values)
    
    def state_dict(self):
        return {'values': self._accumulated_values}

    def load_state_dict(self, state_dict):
        self._accumulated_values = state_dict['values']
    
    def reduce(self):
        return sum(self._accumulated_values) / len(self._accumulated_values)

    def clear(self):
        self._accumulated_values = []


class MetricAccumulator(Callback):
    def __init__(
            self,
            accumulators: dict[str, Accumulator],
            *,
            reset_every_num_steps=None,
        ):
        super().__init__()
        self._accumulators = accumulators
        self._reset_every_num_steps = reset_every_num_steps

    def state_dict(self):
        state_dict = {}
        for idx, accumulator in enumerate(self._accumulators.values()):
            state_dict[idx] = accumulator.state_dict()
    
    def load_state_dict(self, state_dict):
        for idx, accumulator in enumerate(self._accumulators.values()):
            accumulator.load_state_dict(state_dict[idx])
    
    def before_run(self, runner: Runner):
        self.clear()

    def after_step(self, runner: Runner, context: RunnerContext):
        for name, accumulator in self._accumulators.items():
            accumulator.accumulate(context.metrics[name])
        self.reduce(context)
        if self._reset_every_num_steps is not None and runner.global_step % self._reset_every_num_steps == 0:
            self.clear()
    
    def reduce(self, context: RunnerContext):
        for name, accumulator in self._accumulators.items():
            context.metrics[name] = accumulator.reduce()
        
    def after_run(self, runner: Runner, context: RunnerContext):
        if self._reset_every_num_steps is None:
            self.reduce(context)
        self.clear()
    
    def clear(self):
        for accumulator in self._accumulators.values():
            accumulator.clear()


class Validation(TrainingCallback):
    def __init__(
            self, 
            dataset, 
            callbacks,
            *, 
            model=None
        ):
        if hasattr(dataset, '__next__') and not hasattr(dataset, '__getitem__'):
            raise TypeError(f'Dataset expected to be iterable but not iterator, got {type(dataset)}')
        self._dataset = dataset
        self._model = model
        self._callbacks = callbacks

    def after_step(self, runner: TrainingRunner, context: TrainingRunnerContext):
        logger.info('Doing validation')

        inference_result = InferenceRunner(
            model=(self._model if self._model is not None else runner.model),
            dataset=self._dataset,
            callbacks=self._callbacks
        ).run()

        for name, value in inference_result.metrics.items():
            if name in context.metrics:
                raise ValueError(f'Metric already exists: {name}')
            context.metrics[name] = value

