from utils import parse_args
import sys, os, json

def main():
    config, path_to_config, iter_from, iter_to = parse_args()

    for iter in range (iter_from, iter_to+1):
        i = "{:02d}".format(iter_from)
        i = "{}-{}".format(i[:-1],i[-1])

        config['experiment_name'] =  '{}__{}_'.format(config['experiment_name'], i)

        with open(path_to_config, 'w') as config_f:
            json.dump(config, config_f)

        os.system("python3 ../modeling/infer.py --params {}.json".format(path_to_config))



    