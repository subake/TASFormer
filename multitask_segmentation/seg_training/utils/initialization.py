'''
Very important file.
initialize() must be called from train.py

Functions here are responsible for loading data config from .yaml file
'''

import os
from argparse import Namespace
import yaml

import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


def initialize(config: Namespace):
    '''
    1. Set random seeds
    2. Initialize wandb if `config.wandb_project` is given
    3. Renames wandb run if `config.name` is given
    4. Loads parameters from config file if given

    Parameters
    ----------
    config: Namespace

    Returns
    -------
    data_config: dict
    '''
    logger = []

    # Initialize wandb run
    if config.wandb_project is not None:
        logger.append(WandbLogger(
            config.name,
            config=config.__dict__,
            project=config.wandb_project,
            sync_tensorboard=True,
            log_model=False,
        ))

    logger.append(TensorBoardLogger('logs/'))

    # Set the random seeds
    np.random.seed(0)
    torch.manual_seed(0)

    # Set default accelerator to ddp
    if config.gpus is not None and config.gpus > 1 and config.accelerator is None:
        config.accelerator = 'ddp'

    # Load data config from file
    data_config_path = config.data_config or f'configs/datasets[{os.uname().nodename}].yaml'
    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)

    return data_config, logger


def set_argparse_defaults_from_yaml(parser):
    config = parser.parse_args()

    # Read config from config file
    if config.config is not None:
        with open(config.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    return parser