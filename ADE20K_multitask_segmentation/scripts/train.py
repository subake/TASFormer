'''
Training script
'''

import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import matplotlib
matplotlib.use('agg')

import seg_training
import time

def main(config):
    torch.set_num_threads(1)

    # Initialize config
    data_config, logger = seg_training.initialize(config)

    # Create train, valid, and test datasets
    data_module = seg_training.datasets.pl_datamodule.SegDataModule(config, data_config)

    # Create lit module
    if config.load_from is not None:
        lit_module = lit_module = seg_training.pl_module.LitSeg(**config.__dict__)
        lit_module.load_state_dict(torch.load(config.load_from)["state_dict"], strict=False)
    else:
        lit_module = seg_training.pl_module.LitSeg(**config.__dict__)

    # Create trainer
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor='ade_val/iou',
        mode='max',
        dirpath='./',
        filename=f'model_{config.name}' + '_-epoch-{epoch:02d}-iou-{ade_val/iou:.4f}',
        auto_insert_metric_name=False)

    trainer = pl.Trainer.from_argparse_args(
        config,
        logger=logger,
        multiple_trainloader_mode='max_size_cycle',
        callbacks=[checkpoint_callback],
    )

    # Train model
    trainer.fit(
        lit_module,
        data_module
    )

    # Print sample output
    lit_module.eval()
    torch.manual_seed(0)
    sample_image = torch.randn(
        1, 3, config.image_h, config.image_w, requires_grad=True)
    torch_out = lit_module(sample_image)

    # Save model (with optimizer and scheduler for future)
    trainer.save_checkpoint(f'model_{config.name}.ckpt')

    # Test
    if len(config.test_datasets) > 0:
        trainer.test(
            lit_module,
            datamodule=data_module,
        )


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train 1D image segmentation')

    parser.add_argument(
        '--config', type=str,
        help='configuration file in yaml format (ex.: configs/config_warehouse.yaml)')
    parser.add_argument(
        '--data_config', type=str, default=None,
        help='''
            dataset configuration file in yaml format.
            Will be set by default to datasets/datasets["machine name"].yaml
        ''')
    parser.add_argument(
        '--name', type=str, default=None,
        help='name of the experiment for logging')
    parser.add_argument(
        '--wandb_project', type=str, default=None,
        help='project name for Weights&Biases')
    parser.add_argument(
        '--load_from', type=str, default=None,
        help='saved checkpoint')


    parser = seg_training.pl_module.LitSeg.add_argparse_args(parser)
    parser = seg_training.datasets.pl_datamodule.SegDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser = seg_training.utils.initialization.set_argparse_defaults_from_yaml(parser)

    args = parser.parse_args()

    if args.name is None:
        raise BaseException(
            '''Argument --name is not specified. If it is specified but you see this error,
check out this bug https://github.com/matplotlib/matplotlib/issues/17379'''
        )

    # Run
    main(args)
