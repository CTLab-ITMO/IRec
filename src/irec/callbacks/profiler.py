import torch
from loguru import logger

from irec.callbacks.base import Callback
from irec.runners.base import Runner, RunnerContext


class Profiler(Callback):
    def __init__(self, wait, warmup, active, logdir, worker_idx=0):
        assert wait + warmup > 0, 'Should have atleast some warmup before profiling'
        self._wait = wait
        self._warmup = warmup
        self._active = active
        self._logdir = logdir
        self._worker_idx = worker_idx
        self._curr_step = 0
        self._profiler = None

    def before_run(self, runner: Runner):
        if self._profiler is None:
            logger.info('Creating profiler')
            self._profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=self._wait,
                    warmup=self._warmup,
                    active=self._active
                ),
                record_shapes=True,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self._logdir, worker_name=f'worker{self._worker_idx}'),
                profile_memory=True
            )
            self._profiler.start()

    def after_step(self, runne: Runner, context: RunnerContext):
        if self._profiler is not None:
            self._profiler.step()
            if self._curr_step > self._wait + self._warmup + self._active:
                self._profiler.stop()
                self._profiler = None
            self._curr_step += 1
