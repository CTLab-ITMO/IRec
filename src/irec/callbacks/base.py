import functools
import inspect
import types

from functools import cached_property
from typing import Any, Callable, Union


class Callback:
    """
        Runners have an ability to trigger different methods during execution.

        Context presumes that there should be some callbacks that work before the current one and produce necessary input.
        Runner state is necessary for some callbacks and represents literal runner state and step state

        Typical hooks include:
        * before_run
        * load_snapshot
            * runner step with specific callbacks
            * after_step
            * save_snapshot
        * save_snapshot
        * after_run
    """
    def state_dict(self):
        """ Callback state_dict is used for snapshotting / checkpointing. """
        return {}

    def load_state_dict(self, state_dict):
        """ Should be able to load the state_dict it provided with method .state_dict """
        pass

    def save_snapshot(self, runner):
        pass

    def load_snapshot(self, runner):
        pass

    def before_run(self, runner):
        pass

    def after_step(self, runner, context):
        pass

    def after_run(self, runner, context):
        pass

    declared_events = frozenset({before_run, load_snapshot, after_step, save_snapshot, after_run})

    @cached_property
    def implemented_events(self):
        return frozenset({
            event
            for event in self.declared_events
            if getattr(self, event.__name__) != types.MethodType(event, self)
        })

    def every_num_steps(self, num_steps):
        return EveryNumSteps(self, num_steps)

    def ignore_if(self, predicate):
        return Callback() if predicate else self


class BatchCallback(Callback):
    def before_load(self, runner, context):
        pass

    def before_process_batch(self, runner, context):
        pass

    declared_events = Callback.declared_events | frozenset({before_load, before_process_batch})


class Composite(Callback):
    def __init__(self, *callbacks, declared_events=None):
        super().__init__()
        self._callbacks = callbacks
        self._declared_events = frozenset({event for callback in self._callbacks for event in callback.declared_events})
        if declared_events is not None:
            self._declared_events = frozenset(declared_events)
            if not self.implemented_events.issubset(self._declared_events):
                raise TypeError(f'Not declared events were found in callbacks: {self.implemented_events - self._declared_events}')
        for event in self._declared_events:
            if hasattr(self, event.__name__) and event not in Callback.declared_events:
                raise TypeError(f'Event {event} conflict in CompositeCallback')
            setattr(self, event.__name__, functools.partial(self._emit, event))

    @property
    def callbacks(self):
        return self._callbacks

    def __len__(self):
        return len(self._callbacks)

    def __getitem__(self, index):
        return self._callbacks[index]

    def state_dict(self):
        return [callback.state_dict() for callback in self._callbacks]

    def load_state_dict(self, state_dict):
        for callback, state in zip(self._callbacks, state_dict):
            callback.load_state_dict(state)

    @property
    def declared_events(self):
        return self._declared_events

    @cached_property
    def implemented_events(self):
        return frozenset({event for callback in self._callbacks for event in callback.implemented_events})

    def _emit(self, event, runner, *args, **kwargs):
        for callback in self._callbacks:
            if event in callback.declared_events:
                getattr(callback, event.__name__)(runner, *args, **kwargs)


class EveryNumSteps(Callback):
    def __init__(self, callback, num_steps):
        super().__init__()
        self._callback = callback
        self._num_steps = num_steps
        for event in self.declared_events:
            if hasattr(self, event.__name__) and event not in Callback.declared_events:
                raise TypeError(f'Event {event} conflict in EveryNumSteps')
            setattr(self, event.__name__, functools.partial(self._emit, event))

    @property
    def callback(self):
        return self._callback

    def state_dict(self):
        return self._callback.state_dict()

    def load_state_dict(self, state_dict):
        return self._callback.load_state_dict(state_dict)

    @property
    def declared_events(self):
        return self._callback.declared_events

    @property
    def implemented_events(self):
        return self._callback.implemented_events

    def _emit(self, event, runner, *args, **kwargs):
        if runner.global_step % self._num_steps == 0 or runner.global_finished:
            getattr(self._callback, event.__name__)(runner, *args, **kwargs)


class LambdaCallback(Callback):
    def __init__(self, function: Union[Callable[[], Any], Callable[..., Any]]):
        super().__init__()
        self._function = function
        self._has_args = (len(inspect.signature(function).parameters) > 0)
        self._check_signature(inspect.signature(function if self._has_args else self._emit))  # TODO try remove if else and pass `self._emit`

    def _emit(self, *args, **kwargs):
        return self._function(*args, **kwargs) if self._has_args else self._function()

    def _check_signature(self, signature):
        pass


class SaveSnapshot(LambdaCallback):
    def save_snapshot(self, runner):
        self._emit(runner)

    def _check_signature(self, signature):
        signature.bind(None)


class LoadSnapshot(LambdaCallback):
    def load_snapshot(self, runner):
        self._emit(runner)

    def _check_signature(self, signature):
        signature.bind(None)


class BeforeRun(LambdaCallback):
    def before_run(self, runner):
        self._emit(runner)

    def _check_signature(self, signature):
        signature.bind(None)


class BeforeLoad(BatchCallback, LambdaCallback):
    def before_load(self, runner, context):
        self._emit(runner, context)

    def _check_signature(self, signature):
        signature.bind(None, None)


class BeforeBatch(BatchCallback, LambdaCallback):
    def before_batch(self, runner, context):
        self._emit(runner, context)

    def _check_signature(self, signature):
        signature.bind(None, None)


class AfterStep(LambdaCallback):
    def after_step(self, runner, context):
        self._emit(runner, context)

    def _check_signature(self, signature):
        signature.bind(None, None)


class AfterRun(LambdaCallback):
    def after_run(self, runner, context):
        self._emit(runner, context)

    def _check_signature(self, signature):
        signature.bind(None, None)

