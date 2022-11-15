from utils import parse_args, create_logger, fix_random_seed, DEVICE

from dataset import BaseDataset
from dataloader import BaseDataloader
from models import BaseModel
from loss import BaseLoss
from optimizer import BaseOptimizer
from callbacks import BaseCallback

import json
import torch

logger = create_logger(name=__name__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed_val = 42
epoch_cnt = 10


def train(dataloader, model, loss_function, optimizer, callback, epoch_cnt):
    step_num = 0

    for epoch in range(epoch_cnt):
        logger.debug(f'Start epoch {epoch}')
        for step, inputs in enumerate(dataloader):
            model.train()

            for key, values in inputs.items():
                inputs[key] = inputs[key].to(device)

            result = model(inputs)
            result = loss_function(result)

            optimizer.step(result)
            callback(result, step_num)
            step_num += 1

    logger.debug('Training procedure has been finished!')


def main():
    fix_random_seed(seed_val)
    config = parse_args()

    logger.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))

    dataset = BaseDataset.create_from_config(config['dataset'])

    train_dataset, validation_dataset, _ = dataset.get_samplers()

    train_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['train'],
        dataset=train_dataset
    )

    validation_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'],
        dataset=validation_dataset
    )

    model = BaseModel.create_from_config(
        config['model'],
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        max_sequence_len=dataset.max_sequence_length
    ).to(DEVICE)

    loss_function = BaseLoss.create_from_config(config['loss'])
    optimizer = BaseOptimizer.create_from_config(config['optimizer'], model=model)

    callback = BaseCallback.create_from_config(
        config['callback'],
        model=model,
        dataloader=validation_dataloader,
        optimizer=optimizer
    )

    # TODO add verbose option for all callbacks, multiple optimizer options (???), create strong baseline
    # TODO create pre/post callbacks
    logger.debug('Everything is ready for training process!')
    logger.debug('Start training...')

    # TODO check the convergence and overall code
    # Train process
    train(
        dataloader=train_dataloader,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        callback=callback,
        epoch_cnt=config['train_epochs_num']
    )


if __name__ == '__main__':
    main()
