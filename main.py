from modeling import BaseModel, BaseDataset

import json
import argparse

import logging
logger = logging.getLogger(__name__)


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    args = parser.parse_args()
    with open(args.params) as f:
        params = json.load(f)
    return params


if __name__ == '__main__':
    params = parse_params()
    model = BaseModel.create_from_config(params['model'])
    dataset = BaseDataset.create_from_config(params['dataset'])
