import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from typing import Optional
import torch
from torch.utils.data import DataLoader
import albumentations as A
import pytorch_lightning as pl

import nnio
import re
import json
import os, sys

from PIL import Image

class TranspDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path,
            image_h,
            image_w,    
            categories,
            cat_mapping,
            augmentations=None,
            split=None,
    ):
        """
        Parameters
        ----------
        path (str): list of paths to sequences of left frames
        augmentations: albumentations augmentations
        """

        super().__init__()

        self.augmentations = augmentations

        self.path = path

        self.image_h = image_h
        
        self.image_w = image_w

        self.categories = categories
        
        self.cat_mapping = cat_mapping
        
        self.split = split

        self.img_paths = []
        
        self.mask_paths = []

        self.preproc = nnio.Preprocessing(
            dtype='float32',
            channels_first=True,
            divide_by_255=True,
        )
        initial_classes = [
            'background',
            'shelf',
            'jar or tank',
            'freezer',
            'window',
            'glass door',
            'eyeglass',
            'cup',
            'floor glass',
            'glass bowl',
            'water bottle',
            'storage box'
        ]

        # Remap transp to new classes
        cat2id = {
            self.categories[i]: i
            for i in range(len(self.categories))
        }
        for key in self.cat_mapping:
            cat2id[key] = cat2id[self.cat_mapping[key]]
        
        self.transp_to_classid = {}
        for cat in initial_classes:
            idx = initial_classes.index(cat)
            self.transp_to_classid[idx] = cat2id[cat]

        self.images, self.masks = self._get_trans10k_pairs()

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def __getitem__(self, x):
        item, category_idx = x
        img = np.array(Image.open(self.images[item]).convert('RGB'))
        mask = np.array(Image.open(self.masks[item]).convert("P")).astype('int64')
        
        img = cv2.resize(img, dsize=(self.image_h, self.image_w))
        mask = cv2.resize(mask, dsize=(self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)

        for c in self.transp_to_classid:
            mask[mask == c] = self.transp_to_classid[c]
            
        mask[mask != category_idx] *= 0
        mask[mask == category_idx] = 1

        if self.augmentations is not None:
            aug = self.augmentations(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
                
        # Resize, scale, etc
        img = self.preproc(img)
        return {'image': img, 'mask': mask, 'task': category_idx}

    def __len__(self):
        return len(self.img_paths) * (len(self.categories) - 1)

    def _get_trans10k_pairs(self):
        if self.split == 'train':
            img_folder = os.path.join(self.path, 'train/images')
            mask_folder = os.path.join(self.path, 'train/masks_12')
        elif self.split == "val":
            img_folder = os.path.join(self.path, 'validation/images')
            mask_folder = os.path.join(self.path, 'validation/masks_12')
        else:
            self.split == "test"
            img_folder = os.path.join(self.path, 'test/images')
            mask_folder = os.path.join(self.path, 'test/masks_12')

        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '_mask.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    self.img_paths.append(imgpath)
                    self.mask_paths.append(maskpath)
                else:
                    logging.info('cannot find the mask:', maskpath)

        return self.img_paths, self.mask_paths