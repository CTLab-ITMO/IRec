import json
import numpy as np
import torch
import pickle

from irec.callbacks.train import TrainingCallback
from irec.runners.train import TrainingRunner, TrainingRunnerContext


class InferenceSaver(TrainingCallback):
    def __init__(self, metrics, save_path, format='pickle'):
        super().__init__()
        self._metrics = metrics
        self._save_path = save_path
        self._format = format
        self._accumulated_result = list()
        assert format in ['pickle', 'json'], 'Unknown inference format!'

    def state_dict(self):
        return {'accumulate_result': self._accumulated_result}

    def load_state_dict(self, state_dict):
        self._accumulated_result = state_dict['accumulate_result']

    def before_run(self, runner: TrainingRunner):
        return super().before_run(runner)

    def after_step(self, runner: TrainingRunner, context: TrainingRunnerContext):
        batch_result = self._metrics(context.batch, context.model_outputs, context.metrics)
        processed_batch_result = {}
        for key, values in batch_result.items():
            if isinstance(values, torch.Tensor):
                processed_batch_result[key] = values.tolist()
            elif isinstance(values, np.ndarray):
                processed_batch_result[key] = values.tolist()
            else:
                assert isinstance(values, list)
                processed_batch_result[key] = values

        self._accumulated_result.extend([
            {key: values[i] for key, values in processed_batch_result.items()}
            for i in range(len(next(iter(processed_batch_result.values()))))
        ])

    def after_run(self, runner: TrainingRunner, context: TrainingRunnerContext):
        if self._format == 'pickle':
            with open(self._save_path, 'wb') as f:
                pickle.dump(self._accumulated_result, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self._format == 'json':
            with open(self._save_path, 'w') as f:
                json.dump(self._accumulated_result, f, indent=2)
