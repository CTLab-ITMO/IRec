from utils import parse_args
from data import BaseDataset
from model import BaseModel
from loss import BaseLoss
from metric import BaseMetric

import random
import numpy as np

import torch

from torch.utils.data import DataLoader

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def main():
    config = parse_args()

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

    print(config)
    model = BaseModel.create_from_config(config['model']).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    loss = BaseLoss.create_from_config(config['loss'])
    metrics = BaseMetric.create_from_config(config['metrics'])
    # TODO Put loss and metrics into model creation

    for epoch in range(train_epochs_num):
        for step, inputs in enumerate(train_dataloader):
            # Put data on the device (cpu/gpu)
            for key, values in inputs.items():
                inputs[key] = torch.squeeze(inputs[key]).to(device)

            # Forward step
            result = model(inputs)

            print(result)

            # Compute loss
            loss_value = loss(predict=result, ground_truth=inputs)

            # Compute metrics
            metric_values = metrics(predict=result, ground_truth=inputs)

            # Compute backward step
            optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # TODO do validation if needed
            # TODO callbacks


if __name__ == '__main__':
    main()
