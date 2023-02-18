import utils
from utils import parse_args, create_logger, fix_random_seed, DEVICE, Params, dict_to_str
from train import train


from dataset import BaseDataset
from dataloader import BaseDataloader
from models import BaseModel
from optimizer import BaseOptimizer
from loss import BaseLoss
from callbacks import BaseCallback

import json
import torch

logger = create_logger(name=__name__)
# seed_val = 42


def main():
    # fix_random_seed(seed_val)
    config = parse_args()

    logger.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))

    dataset_params = Params(config['dataset'], config['dataset_params'])
    model_params = Params(config['model'], config['model_params'])
    loss_function_params = Params(config['loss'], config['loss_params'])
    optimizer_params = Params(config['optimizer'], config['optimizer_params'])

    logger.debug('Everything is ready for training process!')

    start_from = config.get('start_from', 0)
    cnt = 0

    for dataset_param in dataset_params:
        for model_param in model_params:
            for loss_param in loss_function_params:
                for optimizer_param in optimizer_params:
                    cnt += 1
                    if cnt < start_from:
                        continue

                    model_name = '_'.join([
                        config['experiment_name'],
                        dict_to_str(dataset_param, config['model_params']),
                        dict_to_str(model_param, config['model_params']),
                        dict_to_str(loss_param, config['loss_params']),
                        dict_to_str(optimizer_param, config['optimizer_params'])
                    ])

                    logger.debug('Starting {}'.format(model_name))

                    dataset = BaseDataset.create_from_config(dataset_param)

                    train_sampler, test_sampler = dataset.get_samplers()

                    train_dataloader = BaseDataloader.create_from_config(
                        config['dataloader']['train'],
                        dataset=train_sampler,
                        **dataset.meta
                    )

                    validation_dataloader = BaseDataloader.create_from_config(
                        config['dataloader']['validation'],
                        dataset=test_sampler,
                        **dataset.meta
                    )

                    if utils.tensorboards.GLOBAL_TENSORBOARD_WRITER is not None:
                        utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.close()
                    utils.tensorboards.GLOBAL_TENSORBOARD_WRITER = utils.tensorboards.TensorboardWriter(model_name)

                    model = BaseModel.create_from_config(model_param, **dataset.meta).to(DEVICE)
                    loss_function = BaseLoss.create_from_config(loss_param)
                    optimizer = BaseOptimizer.create_from_config(optimizer_param, model=model)

                    callback = BaseCallback.create_from_config(
                        config['callback'],
                        model=model,
                        dataloader=validation_dataloader,
                        optimizer=optimizer
                    )

                    best_model_checkpoint = train(
                        dataloader=train_dataloader,
                        model=model,
                        optimizer=optimizer,
                        loss_function=loss_function,
                        callback=callback,
                        epoch_cnt=config['train_epochs_num']
                    )

                    logger.debug('Saving best model checkpoint...')
                    checkpoint_path = '../checkpoints/{}_best_checkpoint.pth'.format(model_name)
                    torch.save(best_model_checkpoint, checkpoint_path)
                    logger.debug('Saved model as {}'.format(checkpoint_path))


if __name__ == '__main__':
    main()
