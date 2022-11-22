from utils import parse_args, create_logger, fix_random_seed, DEVICE

from dataset import BaseDataset
from dataloader import BaseDataloader
from models import BaseModel
from metric import BaseMetric

import json
import torch
from collections import Counter


logger = create_logger(name=__name__)
seed_val = 42


def inference(dataloader, model, metrics, pred_prefix, labels_prefix):
    model.eval()
    running_metrics = Counter()

    with torch.no_grad():
        for batch in dataloader:
            for key, values in batch.items():
                batch[key] = batch[key].to(DEVICE)

            _, batch = model(batch)

            for key, values in batch.items():
                batch[key] = batch[key].cpu()

            for metric_name, metric_function in metrics.items():
                running_metrics[metric_name] += metric_function(
                    inputs=batch,
                    pred_prefix=pred_prefix,
                    labels_prefix=labels_prefix,
                )

    logger.debug('Inference procedure has been finished!')
    logger.debug('Metrics are the following:')
    for metric_name, metric_values in running_metrics.items():
        logger.info('{}: {}'.format(metric_name, sum(metric_values) / len(metric_values)))


def main():
    fix_random_seed(seed_val)
    config = parse_args()

    logger.debug('Inference config: \n{}'.format(json.dumps(config, indent=2)))

    dataset = BaseDataset.create_from_config(config['dataset'])

    _, _, eval_dataset = dataset.get_samplers()

    eval_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'],
        dataset=eval_dataset
    )

    model = BaseModel.create_from_config(
        config['model'],
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        max_sequence_len=dataset.max_sequence_length
    ).to(DEVICE)

    checkpoint_path = '../checkpoints/{}_final_state.pth'.format(config['experiment_name'])
    model.load_state_dict(torch.load(checkpoint_path))

    metrics = {
        metric_name: BaseMetric.create_from_config(metric_cfg)
        for metric_name, metric_cfg in config['metrics'].items()
    }

    inference(eval_dataloader, model, metrics, config['pred_prefix'], config['labels_prefix'])


if __name__ == '__main__':
    main()
