import os, json, itertools
from utils import parse_args, create_logger, Params, dict_to_str

logger = create_logger(name=__name__)


def main():
    logger.debug('Multiple Inference procedure has been started...')
    
    # config, path_to_config, iter_from, iter_to = parse_args()
    config, path_to_config, train_multiple_config = parse_args()
    logger.debug('...arguments parsed...')

    # load train_grid params from related train_multiple config
    dataset_params = Params(train_multiple_config['dataset'], train_multiple_config['dataset_params'])
    model_params = Params(train_multiple_config['model'], train_multiple_config['model_params'])
    loss_function_params = Params(train_multiple_config['loss'], train_multiple_config['loss_params'])
    optimizer_params = Params(train_multiple_config['optimizer'], train_multiple_config['optimizer_params'])

    list_of_params = list(itertools.product(
        dataset_params,
        model_params,
        loss_function_params,
        optimizer_params
    ))
    
    # for iter in range (iter_from, iter_to+1):
    for iter in range(len(list_of_params)):
        logger.debug('...iteration {} of {}...'.format(iter, len(list_of_params)-1))
        
        # # change experiment_name to get right checkpoint
        # i = "{:02d}".format(iter)
        # i = "{}-{}".format(i[:-1],i[-1])
        # experiment_name = '_'.join(config['experiment_name'].split('_')[:4])
        # config['experiment_name'] = '{}__{}__'.format(experiment_name, i)

        # change model hyperparameters according to train_grid and checkpoint
        dataset_param, model_param, loss_param, optimizer_param = list_of_params[iter]
        config['dataset'] = dataset_param
        config['model'] = model_param

        # change experiment_name to get right checkpoint
        config['experiment_name'] = '_'.join([
            '{}grid'.format(config['experiment_name'].split('grid')[0]),
            dict_to_str(dataset_param, train_multiple_config['dataset_params']),
            dict_to_str(model_param, train_multiple_config['model_params']),
            dict_to_str(loss_param, train_multiple_config['loss_params']),
            dict_to_str(optimizer_param, train_multiple_config['optimizer_params'])
        ])

        logger.debug('Starting {} inference'.format(config['experiment_name']))

        # save new infer_config, related to the given checkpoint
        with open(path_to_config, 'w') as config_f:
            json.dump(config, config_f)

        subprocess_result = os.system("python3 ./infer.py --params {}".format(path_to_config))
        logger.debug('SUBPROCESS RESULT: {}'.format(subprocess_result))
        if subprocess_result != 0:
            logger.debug('EXITING')
        
    
    logger.debug('Multiple Inference procedure has been finished!')


if __name__ == '__main__':
    main()
