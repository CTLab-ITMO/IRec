import time

from irec.callbacks.base import BatchCallback, Callback
from irec.runners.base import BatchRunner, BatchRunnerContext, Runner, RunnerContext


class CpuTimer:
    #TODO: implement CudaTimer
    def __init__(self):
        super().__init__()
        self._start = None

    def start(self):
        assert self._start is None
        self._start = time.perf_counter()

    def stop(self):
        assert self._start is not None
        result = (time.perf_counter() - self._start) * 1000.
        self._start = None
        return result


class MeasureStepTime(BatchCallback):
    def __init__(self, name='time/step'):
        super().__init__()
        self._name = name
        self._timer = CpuTimer()

    def before_batch(self, runner: BatchRunner, context: BatchRunnerContext):
        self._timer.start()

    def after_step(self, runner: BatchRunner, context: BatchRunnerContext):
        context.metrics[self._name] = self._timer.stop()


class MeasureLoadingTime(BatchCallback):
    def __init__(self, name='time/load'):
        super().__init__()
        self._name = name
        self._timer = CpuTimer()

    def before_load(self, runner: BatchRunner, context: BatchRunnerContext):
        self._timer.start()

    def before_batch(self, runner: BatchRunner, context: BatchRunnerContext):
        context.metrics[self._name] = self._timer.stop()


class MeasureTotalStepTime(Callback):
    def __init__(self, name='time/total'):
        super().__init__()
        self._name = name
        self._timer = CpuTimer()

    def before_run(self, runner: Runner):
        self._timer.start()

    def after_step(self, runner: Runner, context: RunnerContext):
        context.metrics[self._name] = self._timer.stop()
        self._timer.start()

    def after_run(self, runner: Runner, context: RunnerContext):
        self._timer.stop()
