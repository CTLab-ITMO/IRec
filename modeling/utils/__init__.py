from .registry import MetaParent
from .tensorboards import GLOBAL_TENSORBOARD_WRITER

import json
import random
import logging
import argparse
import numpy as np

import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    args = parser.parse_args()
    with open(args.params) as f:
        params = json.load(f)
    return params


def create_logger(
        name,
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
):
    logging.basicConfig(level=level, format=format, datefmt=datefmt)
    logger = logging.getLogger(name)
    return logger


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_to_list(values):
    if not isinstance(values, list):
        values = [values]
    return values
