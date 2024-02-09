from utils import parse_args, create_logger, fix_random_seed, DEVICE

from dataset import BaseDataset
from dataloader import BaseDataloader
from models import BaseModel, TorchModel
from metric import BaseMetric

import json
import numpy as np
import torch
import datetime


logger = create_logger(name=__name__)
seed_val = 42


def inference(dataloader, model, metrics, pred_prefix, labels_prefix, output_path=None, output_params=None):
    running_metrics = {}
    for metric_name, metric_function in metrics.items():
        running_metrics[metric_name] = []

    if isinstance(model, TorchModel):
        model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            print(idx, len(running_metrics['ndcg@20']))

            for key, value in batch.items():
                batch[key] = value.to(DEVICE)
            batch[pred_prefix] = model(batch)

            for key, values in batch.items():
                batch[key] = values.cpu()

            for metric_name, metric_function in metrics.items():
                running_metrics[metric_name].extend(metric_function(
                    inputs=batch,
                    pred_prefix=pred_prefix,
                    labels_prefix=labels_prefix,
                ))

    logger.debug('Inference procedure has been finished!')
    logger.debug('Metrics are the following:')
    for metric_name, metric_value in running_metrics.items():
        logger.info('{}: {}'.format(metric_name, np.mean(metric_value)))

    #TODO implement output_path as argument in utils.parse_args
    #TODO add other output_params if needed
    if output_path:
		experiment_name = output_params['experiment_name'].replace('light_gcn','lightgcn')
        line = {
            'datetime': str(datetime.datetime.now().replace(microsecond=0)),
            'experiment_name': output_params['experiment_name'],
            'model': experiment_name.split('_')[0].replace('lightgcn','light_gcn'),
            'dataset': experiment_name.split('_')[1],
            'domain': experiment_name.split('_')[2]
        }
        for metric_name, metric_value in running_metrics.items():
            line[metric_name] = round(np.mean(metric_value), 18)

        with open(output_path, 'a') as output_file:
            output_file.write('{}\n'.format(json.dumps(line)))


def main():
    fix_random_seed(seed_val)
    config = parse_args()
    #TODO implement output_path as argument in utils.parse_args
    #TODO add other output_params if needed
    output_path = '../checkpoints/metrics.log'
    output_params = {'experiment_name': config['experiment_name']}

    logger.debug('Inference config: \n{}'.format(json.dumps(config, indent=2)))

    dataset = BaseDataset.create_from_config(config['dataset'])

    _, _, eval_dataset = dataset.get_samplers()

    eval_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'],
        dataset=eval_dataset
    )

    model = BaseModel.create_from_config(config['model'], **dataset.meta)

    if isinstance(model, TorchModel):
        model = model.to(DEVICE)
        checkpoint_path = '../checkpoints/{}_final_state.pth'.format(config['experiment_name'])
        model.load_state_dict(torch.load(checkpoint_path))

    metrics = {
        metric_name: BaseMetric.create_from_config(metric_cfg)
        for metric_name, metric_cfg in config['metrics'].items()
    }

    if output_path:
        inference(eval_dataloader, model, metrics, config['pred_prefix'], config['label_prefix'], output_path, output_params)
    else:
        inference(eval_dataloader, model, metrics, config['pred_prefix'], config['label_prefix'])


if __name__ == '__main__':
    main()
