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

from . import scenenet_pb2 as sn

class SUNDataset(torch.utils.data.Dataset):
    '''
    Loads images from folders
    '''

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

        self.image_paths = []

        self.masks_path = []

        self.preproc = nnio.Preprocessing(
            dtype='float32',
            channels_first=True,
            divide_by_255=True,
        )
        initial_classes = [
            'wall',
            'floor',
            'cabinet',
            'bed',
            'chair',
            'sofa',
            'table',
            'door',
            'window',
            'bookshelf',
            'picture',
            'counter',
            'blinds',
            'desk',
            'shelves',
            'curtain',
            'dresser',
            'pillow',
            'mirror',
            'floor_mat',
            'clothes',
            'ceiling',
            'books',
            'fridge',
            'tv',
            'paper',
            'towel',
            'shower_curtain',
            'box',
            'whiteboard',
            'person',
            'night_stand',
            'toilet',
            'sink',
            'lamp',
            'bathtub',
            'bag'
        ]
        # Remap sun to new classes
        cat2id = {
            self.categories[i]: i
            for i in range(len(self.categories))
        }
        for key in self.cat_mapping:
            cat2id[key] = cat2id[self.cat_mapping[key]]
        
        sun_to_classid = {}
        for i in initial_classes:
            idx = initial_classes.index(i)
            sun_to_classid[idx + 1] = cat2id[i]

        self.SUN_TO_CLASS = sun_to_classid
        
        images_folder = os.path.join(path, 'images')
        masks_folder = os.path.join(path, 'labels')

        # Get images & masks paths
        for fname in sorted(os.listdir(images_folder)):
            fname_image = re.split(r'.jpg', fname)[0]
            fname_mask = int(re.search(r'[1-9]+[0-9]*', fname).group(0)) + 5050

            img_path = os.path.join(images_folder, fname)
            if self.split == 'train':
                mask_path = os.path.join(masks_folder, "img-"+'{0:06d}'.format(fname_mask) + ".png")
            else:
                mask_path = os.path.join(masks_folder, fname_image + ".png")
            self.image_paths.append(img_path)
            self.masks_path.append(mask_path)


    def __getitem__(self, item):
        # Load image
        image = np.array(Image.open(self.image_paths[item]))
        mask = np.array(Image.open(self.masks_path[item]))

        # Resize image, mask
        image = cv2.resize(image, dsize=(self.image_h, self.image_w))
        mask = cv2.resize(mask, dsize=(self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)

        for c in self.SUN_TO_CLASS:
            mask[mask == c] = self.SUN_TO_CLASS[c]

        # Augment image
        if self.augmentations is not None:
            aug = self.augmentations(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        # Resize, scale, etc
        image = self.preproc(image)
        return {'image': image, 'mask': mask}

    def __len__(self):
        return len(self.image_paths)
