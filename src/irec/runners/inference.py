import dataclasses
from typing import Any

import torch

from irec.runners.base import BatchRunner, BatchRunnerContext


@dataclasses.dataclass
class InferenceRunnerContext(BatchRunnerContext):
    model_outputs: Any = None


class InferenceRunner(BatchRunner):
    def __init__(
            self, 
            model: torch.nn.Module,
            dataset,
            callbacks
    ):
        super().__init__(
            dataset=dataset,
            callbacks=callbacks
        )
        self._model = model

    @property
    def model(self):
        return self._model

    def run(self):
        with torch.inference_mode(mode=True):
            training = self._model.training
            try:
                self._model.eval()
                return super().run()
            finally:
                self._model.train(training)

    def _process_batch(self, context: InferenceRunnerContext):
        _, context.model_outputs = self._model(context.batch)
    
    def _create_context(self):
        return InferenceRunnerContext()

