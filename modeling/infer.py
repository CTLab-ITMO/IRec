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

    if output_path is not None:
        line = {
            'datetime': str(datetime.datetime.now().replace(microsecond=0))
        }
        
        if output_params is not None:
            line.update(output_params)
        
        for metric_name, metric_value in running_metrics.items():
            line[metric_name] = round(np.mean(metric_value), 8)

        with open(output_path, 'a') as output_file:
            output_file.write('{}\n'.format('\t'.join([param_value for param_value in line.values()])))


def main():
    fix_random_seed(seed_val)
    config = parse_args()
    
    output_path = '../checkpoints/metrics.tsv'
    output_params = {
        'experiment_name': config['experiment_name'],
        'model': config['model']['type']
    }
    if 'dataset' not in config['dataset']:
        output_params['dataset'] = config['dataset']['name'].split('/')[0]
        if 'target_domain' in config['dataset']: 
            output_params['domain'] = config['dataset']['target_domain']
        else:
            output_params['domain'] = config['dataset']['name'].split('/')[1]
    else:
        output_params['dataset'] = config['dataset']['dataset']['name'].split('/')[0]
        if 'target_domain' in config['dataset']['dataset']:
            output_params['domain'] = config['dataset']['dataset']['target_domain']
        else:
            output_params['domain'] = config['dataset']['dataset']['name'].split('/')[1]

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

    _ = inference(
        dataloader=eval_dataloader, 
        model=model, 
        metrics=metrics, 
        pred_prefix=config['pred_prefix'], 
        labels_prefix=config['label_prefix'], 
        output_path=output_path, 
        output_params=output_params
    )


if __name__ == '__main__':
    main()
