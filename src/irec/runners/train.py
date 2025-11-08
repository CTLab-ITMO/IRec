import dataclasses
from typing import Any

import torch

import irec.callbacks as cb

from .base import BatchRunner, BatchRunnerContext


@dataclasses.dataclass
class TrainingRunnerContext(BatchRunnerContext):
    model_outputs: Any = None
    training_loss: Any = None


class TrainingRunner(BatchRunner):
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            dataset,
            callbacks
    ):
        super().__init__(
            dataset=dataset,
            callbacks=callbacks
        )
        self._model = model
        self._optimizer = optimizer

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    def state_dict(self):
        state_dict = super().state_dict()
        assert {'model', 'optimizer'}.isdisjoint(state_dict)
        state_dict['model'] = self._model.state_dict()
        state_dict['optimizer'] = self._optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._model.load_state_dict(state_dict['model'], strict=True)
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def run(self):
        assert self._model.training
        self._optimizer.zero_grad()
        return super().run()
    
    @property
    def declared_events(self):
        return cb.TrainingCallback.declared_events

    def _create_context(self):
        return TrainingRunnerContext()
    
    def _process_batch(self, context: TrainingRunnerContext):
        assert self._model.training
        self._run_forward(context)
        self._run_backward(context)
        self._callback.before_optimizer(self, context)
        self._run_optimizer(context)

    def _run_forward(self, context: TrainingRunnerContext):
        context.training_loss, context.model_outputs = self._model(context.batch)

    def _run_backward(self, context: TrainingRunnerContext):
        context.training_loss.backward()

    def _run_optimizer(self, context: TrainingRunnerContext):
        self._optimizer.step()
        self._optimizer.zero_grad()


__all__ = [
    'TrainingRunner',
    'TraninngRunnerContext'
]
