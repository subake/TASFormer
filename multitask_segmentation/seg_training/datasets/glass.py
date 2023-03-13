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

from pycocotools.coco import COCO

import nnio
import re
import json
import os, sys

from PIL import Image

#from . import scenenet_pb2 as sn

class GlassDataset(torch.utils.data.Dataset):
    '''
    Loads images from folders
    '''
    def  __init__(
        self,
        path,
        image_h,
        image_w,
        categories,
        cat_mapping,
        augmentations=None,
    ):
        """
        Parameters
        ----------
        path (str): list of paths to sequences of left frames
        augmentations: albumentations augmentations
        categories: categories of the image's objects
        """

        super().__init__()

        self.augmentations = augmentations

        self.path = path

        self.image_h = image_h
        
        self.image_w = image_w

        self.image_paths = []

        self.masks_path = []

        self.categories = categories
        
        self.cat_mapping = cat_mapping

        initial_classes = [
            'background',
            'glass'
        ]

        # Remap gd to new classes
        cat2id = {
            self.categories[i]: i
            for i in range(len(self.categories))
        }
        for key in self.cat_mapping:
            cat2id[key] = cat2id[self.cat_mapping[key]]
        
        self.gd_to_classid = {}
        for cat in initial_classes:
            idx = initial_classes.index(cat)
            self.gd_to_classid[idx] = cat2id[cat]


        self.preproc = nnio.Preprocessing(
            dtype='float32',
            channels_first=True,
            divide_by_255=True,
        )

        images_folder = os.path.join(path, 'image')
        masks_folder = os.path.join(path, 'mask')

        # Get images & masks paths
        for fname in sorted(os.listdir(masks_folder)):
            fname = re.split(r'.png', fname)[0]
            img_path = os.path.join(images_folder, fname + ".png")
            mask_path = os.path.join(masks_folder, fname + ".png")
            self.image_paths.append(img_path)
            self.masks_path.append(mask_path)

    def __getitem__(self, item):
        # Load image
        image = np.array(Image.open(self.image_paths[item]))
        mask = (np.array(Image.open(self.masks_path[item]))/255)

        # Resize image, mask
        image = cv2.resize(image, dsize=(self.image_h, self.image_w))
        mask = cv2.resize(mask, dsize=(self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)

        for c in self.gd_to_classid:
            mask[mask == c] = self.gd_to_classid[c]

        # Augment image
        if self.augmentations is not None:
            aug = self.augmentations(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
            mask = mask.astype('uint8')
        # Resize, scale, etc
        image = self.preproc(image)
        return {'image': image, 'mask': mask}
    
    def __len__(self):
        return len(self.image_paths)
