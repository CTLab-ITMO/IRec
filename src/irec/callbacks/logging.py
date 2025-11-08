import datetime
import os
import shutil

from loguru import logger

from torch.utils.tensorboard import SummaryWriter

from irec.callbacks.base import Callback
from irec.callbacks.train import TrainingCallback
from irec.runners.base import Runner, RunnerContext
from irec.runners.train import TrainingRunner, TrainingRunnerContext


class LoggingCallback(TrainingCallback):
    def before_run(self, runner: TrainingRunner):
        logger.debug('Before run')
        pass

    def after_step(self, runner: TrainingRunner, context: TrainingRunnerContext):
        logger.debug(f'After step {runner.global_step}')

    def after_run(self, runner: TrainingRunner, context: TrainingRunnerContext):
        logger.debug('After run')

    def before_load(self, runner: TrainingRunner, context: TrainingRunnerContext):
        logger.debug('Before load')

    def before_process_batch(self, runner: TrainingRunner, context: TrainingRunnerContext):
        logger.debug('Before process batch')
    
    def before_optimizer(self, runner: TrainingRunner, context: TrainingRunnerContext):
        logger.debug('Before optimizer')


class TensorboardWriter(SummaryWriter):
    def __init__(
            self,
            experiment_name,
            log_dir,
            use_time=True
        ):
        self._experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        super().__init__(
            log_dir=os.path.join(
                log_dir,
                f'{experiment_name}_{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M" if use_time else "")}'
            )
        )

    def add_scalar(self, *args, **kwargs):
        super().add_scalar(*args, **kwargs)


class TensorboardLogger(Callback):
    def __init__(self, experiment_name, logdir, rewrite=False, step_factor=1):
        super().__init__()
        self._experiment_name = experiment_name
        self._logdir = logdir
        self._rewrite = rewrite
        self._step_factor = step_factor
        self._writer = None
        self._runner = None

    def _setup_writer(self, global_step):
        if self._rewrite and global_step == 1:
            if os.path.exists(self._logdir):
                shutil.rmtree(self._logdir)
        if self._writer is None:
            self._writer = TensorboardWriter(experiment_name=self._experiment_name, log_dir=self._logdir)

    def state_dict(self):
        if self._writer is not None:
            self._writer.flush()
        return {}

    def load_state_dict(self, state_dict):
        self.close_writer()

    def before_run(self, runner):
        if self._runner is None:
            self._runner = runner

    def after_step(self, runner: Runner, context: RunnerContext):
        self._setup_writer(runner.global_step)
        for key, value in context.metrics.items():
            self._writer.add_scalar(key, value, runner.global_step // self._step_factor)

    def after_run(self, runner, context):
        if self._runner is runner:
            self._runner = None
            self.close_writer()

    def close_writer(self):
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


class Logger(Callback):
    def __init__(self, logfile=None, name=None, step_factor=1):
        super().__init__()
        self._logfile = logfile
        # self._writer = logfile if isinstance(logfile, io.IOBase) else None
        self._writer = None
        self._name = name
        self._step_factor = step_factor
        self._runner = None

    def state_dict(self):
        if self._writer is not None:
            self._writer.flush()
        return {}

    def load_state_dict(self, state_dict):
        self.close_writer()

    def before_run(self, runner):
        if self._runner is None:
            self._runner = runner
            logger.info('Starting run')

    def after_step(self, runner: Runner, context: RunnerContext):
        msg = [f'step {runner.global_step // self._step_factor}']
        if self._name is not None:
            msg.insert(0, self._name)
        for key, value in context.metrics.items():
            msg.append(f'{key} {value}')
        if self._logfile is not None:
            if self._writer is None:
                self._writer = open(self._logfile, 'at')
            msg.insert(0, datetime.datetime.now().isoformat(' ', 'milliseconds'))
            self._writer.write(', '.join(msg) + '\n')
            self._writer.flush()
        else:
            logger.info(', '.join(msg))

    def after_run(self, runner, context):
        if self._runner is runner:
            self._runner = None
            self.close_writer()
            logger.info('Finishing run')

    def close_writer(self):
        if self._writer is not None:
            self._writer.flush()
            if self._writer is not self._logfile:
                self._writer.close()
                self._writer = None
