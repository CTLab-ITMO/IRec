from utils import parse_args, create_logger, fix_random_seed

from dataset import BaseDataset
from dataloader import BaseDataloader
from model import BaseModel
from optimizer import BaseOptimizer
from callbacks import BaseCallback

import json
import torch

logger = create_logger(name=__name__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed_val = 42


def train(model, dataloader, optimizer, callback, epoch_cnt):
    step_num = 0

    for epoch in range(epoch_cnt):
        model.train()

        for step, inputs in enumerate(dataloader):
            for key, values in inputs.items():
                inputs[key] = torch.squeeze(inputs[key]).to(device)

            result = model(inputs)
            optimizer.step()
            callback(result, step_num)

            step_num += 1


def main():
    fix_random_seed(seed_val)
    config = parse_args()

    logger.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))

    dataset = BaseDataset.create_from_config(config['dataset'])
    dataloader = BaseDataloader.create_from_config(config['dataloader'], dataset=dataset)
    model = BaseModel.create_from_config(config['model']).to(device)
    optimizer = BaseOptimizer.create_from_config(config['optimizer'], model=model)
    callback = BaseCallback.create_from_config(
        config['callback'],
        model=model,
        dataloader=dataloader,
        optimizer=optimizer
    )

    logger.debug('Everything is ready for training process!')

    # Train process
    train(
        dataloader=dataloader['train'],
        model=model,
        optimizer=optimizer,
        callback=callback,
        epoch_cnt=config['train_epochs_num']
    )


if __name__ == '__main__':
    main()
