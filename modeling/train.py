from utils import parse_args, create_logger, fix_random_seed
from utils import GLOBAL_TENSORBOARD_WRITER

from data import BaseDataset
from model import BaseModel

import json

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed_val = 42


def train(model, dataloader, optimizer, scheduler, epoch_cnt):
    step_num = 0

    for epoch in range(epoch_cnt):
        model.train()
        for step, inputs in enumerate(dataloader):
            for key, values in inputs.items():
                inputs[key] = torch.squeeze(inputs[key]).to(device)

            result = model(inputs)

            # Compute backward step (put everything below in the optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Here it comes to callbacks
            GLOBAL_TENSORBOARD_WRITER.add_scalar('Train/loss', result['loss'], step_num)
            GLOBAL_TENSORBOARD_WRITER.add_scalar('Train/metric/dummy', result['dummy'], step_num)

            step_num += 1

        val_step_num = 0
        model.eval()
        with torch.no_grad():
            mean_loss = 0
            for inputs in dataloader:
                for key, values in inputs.items():
                    inputs[key] = torch.squeeze(inputs[key]).to(device)

                result = model(inputs)
                mean_loss += result['loss']

                val_step_num += 1

            mean_loss /= val_step_num
            GLOBAL_TENSORBOARD_WRITER.add_scalar('Val/loss', mean_loss, step_num)


def main():
    fix_random_seed(seed_val)
    logger = create_logger(name=__name__)

    config = parse_args()
    logger.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))

    dataset = BaseDataset.create_from_config(config['dataset'])
    # train_dataset = dataset.train()
    # val_dataset = dataset.val()

    train_dataloader = DataLoader(
        dataset,  # TODO fix
        batch_size=10,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset,  # TODO fix
        batch_size=10,
        shuffle=False
    )

    train_epochs_num = config['train_epochs_num']
    total_steps = len(train_dataloader) * train_epochs_num
    validation_step_num = config['validation_steps_num']

    model = BaseModel.create_from_config(config['model']).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    train(model, train_dataloader, optimizer, scheduler, 10)

    # for epoch in range(train_epochs_num):
    #     for step, inputs in enumerate(train_dataloader):
    #
    #         # Put data on the device (cpu/gpu)
    #         for key, values in inputs.items():
    #             inputs[key] = torch.squeeze(inputs[key]).to(device)
    #
    #         # Forward step
    #         result = model(inputs)
    #
    #         # Compute loss
    #         loss_value = loss(predict=result, ground_truth=inputs)
    #
    #         # Compute metrics
    #         metric_values = metrics(predict=result, ground_truth=inputs)
    #
    #         # Compute backward step
    #         optimizer.zero_grad()
    #         loss_value.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()
    #         scheduler.step()
    #
    #         step_num += 1
    #
    #
    #         # TODO do validation if needed
    #         # TODO callbacks
    #         # TODO add tensorboard
    #
    #         if step_num % validation_step_num == 0:
    #             # TODO do validation
    #             pass


if __name__ == '__main__':
    main()
