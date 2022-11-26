from utils import parse_args, create_logger, fix_random_seed, DEVICE

from dataset import BaseDataset
from dataloader import BaseDataloader
from models import BaseModel
from optimizer import BaseOptimizer
from loss import BaseLoss
from callbacks import BaseCallback

import json
import torch

logger = create_logger(name=__name__)
seed_val = 42


def train(dataloader, model, optimizer, loss_function, callback, epoch_cnt):
    step_num = 0

    logger.debug('Start training...')

    for epoch in range(epoch_cnt):
        logger.debug(f'Start epoch {epoch}')
        for step, inputs in enumerate(dataloader):
            model.train()

            for key, values in inputs.items():
                inputs[key] = inputs[key].to(DEVICE)

            inputs = model(inputs)

            loss = loss_function(inputs)

            optimizer.step(loss)
            callback(inputs, step_num)
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

    model = BaseModel.create_from_config(config['model'], **dataset.meta).to(DEVICE)

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

    # Train process
    train(
        dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        callback=callback,
        epoch_cnt=config['train_epochs_num']
    )

    logger.debug('Saving model...')
    checkpoint_path = '../checkpoints/{}_final_state.pth'.format(config['experiment_name'])
    torch.save(model.state_dict(), checkpoint_path)
    logger.debug('Saved model as {}'.format(checkpoint_path))


if __name__ == '__main__':
    main()
