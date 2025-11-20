from loguru import logger

import torch


from irec.callbacks.train import TrainingCallback
from irec.runners.train import TrainingRunner


class LoadModel(TrainingCallback):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
    
    def before_run(self, runner: TrainingRunner):
        runner.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        logger.debug(f'Model {self.model_path} is loaded!')
