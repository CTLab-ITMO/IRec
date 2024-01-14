from utils import parse_args, create_logger
import sys, os, json

logger = create_logger(name=__name__)

def main():
    logger.debug('Multiple Inference procedure has been started...')
    
    config, path_to_config, iter_from, iter_to = parse_args()
    logger.debug('...arguments parsed...')
    

    for iter in range (iter_from, iter_to+1):
        logger.debug('...iteration {} of {}...'.format(iter, iter_to))
        
        i = "{:02d}".format(iter)
        i = "{}-{}".format(i[:-1],i[-1])
        experiment_name = '_'.join(config['experiment_name'].split('_')[:4])
        config['experiment_name'] = '{}__{}__'.format(experiment_name, i)

        with open(path_to_config, 'w') as config_f:
            json.dump(config, config_f)

        subprocess_result = os.system("python3 ./infer.py --params {}".format(path_to_config))
        logger.debug('SUBPROCESS RESULT: {}'.format(subprocess_result))
        if subprocess_result == 1:
            logger.debug('EXITING')
        
    
    logger.debug('Multiple Inference procedure has been finished!')


if __name__ == '__main__':
    main()
