import dataclasses
from typing import Any, Dict, Union

import torch

import irec.callbacks as cb


@dataclasses.dataclass
class RunnerContext:
    metrics: Dict[str, Union[int, float]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class BatchRunnerContext(RunnerContext):
    batch: Any = None


class Runner:
    def __init__(self, callbacks: cb.Callback):
        super().__init__()
        self._callback = cb.Composite(*callbacks, declared_events=self.declared_events)
        self._global_step = 0
        self._global_finished = False

    @property
    def callback(self):
        return self._callback

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_finished(self):
        return self._global_finished

    def state_dict(self):
        return {
           'callback': self._callback.state_dict(),
           'global_step': self._global_step,
           'global_finished': self._global_finished
        }

    def load_state_dict(self, state_dict):
        self._callback.load_state_dict(state_dict['callback'])
        self._global_step = state_dict['global_step']
        self._global_finished = state_dict['global_finished']

    def run(self):
        self._callback.before_run(self)
        self._callback.load_snapshot(self)
        while not self._global_finished:
            try:
                context = self._run_step()
                self._callback.after_step(self, context)
                self._callback.save_snapshot(self)
            except StopIteration:
                self._global_finished = True
        context = self._create_context()
        self._callback.save_snapshot(self)
        self._callback.after_run(self, context)
        return context
    
    # Only these two functions below should be re-implemented in other runners
    @property
    def declared_events(self):
        return cb.Callback.declared_events
    
    def _create_context(self):
        return RunnerContext()
    
    def _run_step(self):
        context = self._create_context()
        return context


class BatchRunner(Runner):
    def __init__(self, dataset, callbacks):
        super().__init__(callbacks)
        self._dataset = dataset
        self._dataset_iterator = None

    @property
    def dataset(self):
        return self._dataset

    @property
    def dataset_iterator(self):
        return self._dataset_iterator

    # TODO think
    # @property
    # def dataset_has_state(self):
    #     # TODO: Maybe use runtime checkable Stateful protocol?
    #     # https://pytorch.org/docs/stable/_modules/torch/distributed/checkpoint/stateful.html#Stateful
    #     return (callable(getattr(self._dataset_iterator, 'state_dict', None)) and
    #             callable(getattr(self._dataset_iterator, 'load_state_dict', None)))

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.setdefault('distributed_state_dict', {})
        assert 'dataset' not in state_dict['distributed_state_dict']
        state_dict['distributed_state_dict']['dataset'] = self._dataset_iterator.state_dict() if self.dataset_has_state else None
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.dataset_has_state:
            self._dataset_iterator.load_state_dict(state_dict['distributed_state_dict']['dataset'])

    # TODO I dont'like
    def run(self):
        self._dataset_iterator = iter(self._dataset)
        # self._dataset_iterator = self._dataset
        # self._dataset_iterator = SequenceIterator(self._dataset)
        # Sequential dataset
        # if hasattr(self._dataset, '__getitem__'):
        # # Streaming dataset
        # elif hasattr(self._dataset, '__next__'):
        #     self._dataset_iterator = self._dataset
        # # Generator
        # elif hasattr(self._dataset, '__iter__'):
        #     self._dataset_iterator = iter(self._dataset)
        # else:
        #     raise TypeError(f'Dataset expected to be iterator, iterable or sequence, got {type(self._dataset)}')
        
        return super().run()

    @property
    def declared_events(self):
        return cb.BatchCallback.declared_events
    
    def _create_context(self):
        return BatchRunnerContext()
    
    def _run_step(self):
        context = self._create_context()
        self._global_step += 1
        self._callback.before_load(self, context)
        context.batch = next(self._dataset_iterator)
        self._callback.before_process_batch(self, context)
        self._process_batch(context)
        return context

    def _process_batch(self, context):
        pass


__all__ = [
    'BatchRunner',
    'BatchRunnerContext',
    'Runner',
    'RunnerContext',
]
