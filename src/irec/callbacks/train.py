import torch

from irec.callbacks.base import BatchCallback, LambdaCallback
from irec.runners.train import TrainingRunner, TrainingRunnerContext


class TrainingCallback(BatchCallback):
    def before_optimizer(self, runner: TrainingRunner, context: TrainingRunnerContext):
        pass

    declared_events = BatchCallback.declared_events | frozenset({before_optimizer})


class BeforeOptimizer(TrainingCallback, LambdaCallback):
    def before_optimizer(self, runner, context):
        self._emit(runner, context)

    def _check_signature(self, signature):
        signature.bind(None, None)


class ClipGradient(TrainingCallback):
    def __init__(self, parameters=None, value=1.0, *, name=None):
        self._parameters = list(parameters) if parameters is not None else None
        self._value = value
        self._name = name

    @torch.no_grad()
    def before_optimizer(self, runner: TrainingRunner, context: TrainingRunnerContext):
        if self._parameters is not None:
            norm_before_clip = torch.nn.utils.clip_grad_norm_(self._parameters, self._value)
        elif not hasattr(runner.model, 'clip_grad_norm_'):
            norm_before_clip = torch.nn.utils.clip_grad_norm_(runner.model.parameters(), self._value)
        else:
            norm_before_clip = runner.model.clip_grad_norm_(self._value)
        if self._name is not None:
            if self._name in context.metrics:
                raise ValueError(f'Optimizer name "{self._name}" appears more than once in the list!')
            context.metrics[self._name] = norm_before_clip.item()
