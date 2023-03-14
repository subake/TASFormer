import os
import cv2
import numpy as np
import random
from argparse import Namespace
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import albumentations as A
import pytorch_lightning as pl
import copy

from . import msd
from . import sun
from . import scnet
from . import msd
from . import coco
from . import gta
from . import gdd
from . import transp
from . import metalbot
from . import cityscapes
from . import glass
from . import floor
from . import ade

def reload(category_idx, ds_indices, batch_size):
    indices = [(i, category_idx + 1) for i in ds_indices[category_idx]]
    random.shuffle(indices)
    grouped = [x for x in zip(*[iter(indices)]*batch_size)]
    return grouped
   
    
def train_sampler(ds_indices, batch_size):
    num_categories = len(ds_indices)
    
    max_len = max([len(x) for x in ds_indices])
    categories = np.tile(np.arange(num_categories), int(np.ceil(max_len / batch_size))).tolist()

    indices_by_category = []
    for category_idx in range(num_categories):
        grouped = reload(category_idx, ds_indices, batch_size)
        indices_by_category.append(grouped)

    res = []
    for category_idx in categories:
        if len(indices_by_category[category_idx]) == 0:
            indices_by_category[category_idx] = reload(category_idx, ds_indices, batch_size)
            if len(indices_by_category[category_idx]) == 0:
                continue
        res.append(indices_by_category[category_idx].pop())
    return res


def val_sampler(ds_indices, batch_size):
    num_categories = len(ds_indices)
    
    max_len = max([len(x) for x in ds_indices])
    categories = np.tile(np.arange(num_categories), int(np.ceil(max_len / batch_size))).tolist()

    indices_by_category = []
    for category_idx in range(num_categories):
        grouped = reload(category_idx, ds_indices, batch_size)
        indices_by_category.append(grouped)

    res = []
    for category_idx in categories:
        if len(indices_by_category[category_idx]) == 0:
            continue
        res.append(indices_by_category[category_idx].pop())
    return res

class SegDataModule(pl.LightningDataModule):
    def __init__(self, config: Namespace, data_config: dict):
        super().__init__()

        self.config = config
        self.data_config = data_config

        # Augmentations for training
        self.augmentations = A.Compose([
            # # Random conditions
            # A.RandomShadow(p=0.01),
            # A.RandomRain(p=0.01),
            # A.RandomSnow(p=0.0),
            # A.RandomSunFlare(p=0.01),
            # A.RandomFog(p=0.01),
            # # Simple transforms
            A.RandomBrightnessContrast(p=0.2),
            # A.IAAAdditiveGaussianNoise(p=0.05),
            # A.JpegCompression(p=0.1),
            # # Color transforms
            # A.HueSaturationValue(p=0.5),
            # A.ChannelShuffle(p=0.1),
            # A.ChannelDropout(p=0.05),
            # A.ToGray(p=0.05),
            # Transforms
            # A.Flip(p=0.5),

            A.HorizontalFlip(p=0.5),
            A.augmentations.geometric.transforms.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3)
        ])

        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None
        self.predict_datasets = None

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataloader parameters")
        parser.add_argument('--train_datasets', type=list, default=None)
        parser.add_argument('--dev_datasets', type=list, default=None)
        parser.add_argument('--test_datasets', type=list, default=None)
        parser.add_argument('--predict_datasets', type=list, default=None)
        parser.add_argument('--batch_size', type=int, default=None)
        return parent_parser

    def select_dataset(self, cfg, augmentations=None):
        if cfg['type'] == 'gta':
            return gta.GTA5Dataset(
                path=cfg['path'],
                image_h = self.config.image_h,
                image_w = self.config.image_w,
                augmentations=augmentations,
                categories=self.config.categories,
                cat_mapping=self.config.cat_mapping,
                split=cfg['split'],
            )
        elif cfg['type'] == 'floor':
            return floor.FloorDataset(
                path=cfg['path'],
                image_h = self.config.image_h,
                image_w = self.config.image_w,
                categories=self.config.categories_floor,
                cat_mapping=self.config.cat_mapping_floor,
                augmentations=augmentations,
            )
        elif cfg['type'] == 'glass':
            return glass.GlassDataset(
                path=cfg['path'],
                image_h = self.config.image_h,
                image_w = self.config.image_w,
                categories=self.config.categories_glass,
                cat_mapping=self.config.cat_mapping_glass,
                augmentations=augmentations,
            )
        elif cfg['type'] == 'ade':
            return ade.ADEDataset(
                path=cfg['path'],
                image_h = self.config.image_h,
                image_w = self.config.image_w,
                categories=self.config.categories_ade,
                cat_mapping=self.config.cat_mapping_ade,
                augmentations=augmentations,
                split=cfg['split'],
            )
        elif cfg['type'] == 'msd':
            return msd.MSDDataset(
                path=cfg['path'],
                image_h = self.config.image_h,
                image_w = self.config.image_w,
                categories=self.config.categories_msd,
                cat_mapping=self.config.cat_mapping,
                augmentations=augmentations,
            )
        elif cfg['type'] == 'scnet':
            return scnet.SceneNetDataset(
                path=cfg['path'],
                image_h = self.config.image_h,
                image_w = self.config.image_w,
                protobuf_path=cfg['protobuf_path'],
                path_to_classes=cfg['path_to_classes'],
                categories=self.config.categories_scnet,
                cat_mapping=self.config.cat_mapping_scnet,
                augmentations=augmentations,
            )
        elif cfg['type'] == 'sun':
            return sun.SUNDataset(
                path=cfg['path'],
                image_h = self.config.image_h,
                image_w = self.config.image_w,
                augmentations=augmentations,
                categories=self.config.categories_sun,
                cat_mapping=self.config.cat_mapping_sun,
                split=cfg['split'],
            )
        elif cfg['type'] == 'transp':
            return transp.TranspDataset(
                path=cfg['path'],
                image_h = self.config.image_h,
                image_w = self.config.image_w,
                categories=self.config.categories_trans,
                cat_mapping=self.config.cat_mapping_trans,
                augmentations=augmentations,
                split=cfg['split'],
            )
        elif cfg['type'] == 'metalbot':
            return metalbot.MetalbotDataset(
                path=cfg['path'],
                image_h = self.config.image_h,
                image_w = self.config.image_w,
                augmentations=augmentations,
            )
        elif cfg['type'] == 'city':
            return cityscapes.CityScapesDataset(
                path=cfg['path'],
                augmentations=augmentations,
                categories=self.config.categories,
                image_h = self.config.image_h,
                image_w = self.config.image_w,
                cat_mapping=self.config.cat_mapping,
                split=cfg['split'],
            )
        elif cfg['type'] == 'gdd':
            return gdd.GDDDataset(
                path=cfg['path'],
                augmentations=augmentations,
                categories=self.config.categories_gdd,
                cat_mapping=self.config.cat_mapping_gdd
            )
        else:
            return coco.CocoDataset(
                path=cfg['path'],
                image_h = self.config.image_h,
                image_w = self.config.image_w,
                augmentations=augmentations,                
                categories=self.config.categories,
                cat_mapping=self.config.cat_mapping,
            )

    def setup(self, stage: Optional[str] = None):
        # Create training dataset
        if self.train_dataset is None:
            self.train_datasets = [
                self.select_dataset(
                    self.data_config[ds],
                    augmentations=self.augmentations,
                )
                for ds in self.config.train_datasets
            ]
        # Create validation datasets
        if self.val_datasets is None:
            self.val_datasets = [
                self.select_dataset(
                    self.data_config[ds],
                    augmentations=None,
                )
                for ds in self.config.dev_datasets
            ]

        # Create testing datasets
        if self.test_datasets is None:
            self.test_datasets = [
                self.select_dataset(
                    self.data_config[ds],
                )
                for ds in self.config.test_datasets
            ]

        # Create predict datasets
        if self.predict_datasets is None:
            self.predict_datasets = [
                self.select_dataset(
                    self.data_config[ds],
                )
                for ds in self.config.predict_datasets
            ]        
     
    
    def train_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_sampler=train_sampler(
                        ds.get_ds_indices(),
                        self.config.batch_size
                    ),
                num_workers=4
            )
            for ds in self.train_datasets
        ]
    
    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_sampler=val_sampler(
                        ds.get_ds_indices(),
                        self.config.batch_size
                    ),
                num_workers=4
            )
            for ds in self.val_datasets
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_sampler=val_sampler(
                        ds.get_ds_indices(),
                        self.config.batch_size
                    ),
                num_workers=4
            )
            for ds in self.test_datasets
        ]

    def predict_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=1,
                num_workers=4
            )
            for ds in self.predict_datasets
        ]
