# IRec

## Preparing datasets
Datasets used for experiments:
- Movielens 1M (https://grouplens.org/datasets/movielens):
- Amazon (https://nijianmo.github.io/amazon/index.html):
  - All_Beauty, Clothing_Shoes_and_Jewelry, Sports_and_Outdoors
 To initiate datasets preparation process, one needs to run notebooks from the [notebooks](./notebooks) arhchive.

## Model training
To train a model, simply run the following from the [modeling](./modeling) directory:
```shell
python train.py --params /path/to/the/json/file/with/config
```

The script has 1 input argument: `params` which is the path to the json file with model configuration. The example of such file can be found [here](./configs). This directory contrains json files with model hyperparameters and data preparation instructions. It should contain the following keys:

-`experiment_name` Name of the experiment

-`dataset` Information about the dataset

-`dataloader` Settings for dataloader

-`model` Model hyperparameters

-`optimizer` Optimizer hyperparameters

-`loss` Naming of different loss components

-`callbacks` Different additional traning 

