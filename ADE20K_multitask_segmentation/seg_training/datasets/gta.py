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


class GTA5Dataset(torch.utils.data.Dataset):
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
        split=None,
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

        self.categories = categories
        
        self.cat_mapping = cat_mapping

        self.path = path
        
        self.image_h = image_h
        
        self.image_w = image_w

        self.split = split

        self.image_paths = []

        self.labels_path = []

        self.preproc = nnio.Preprocessing(
            dtype='float32',
            channels_first=True,
            divide_by_255=True,
        )

        initial_classes = [
            'background',
            'ego vehicle', 
            'rectification border', 
            'out of roi', 
            'static',
            'dynamic',
            'ground', 
            'road', 
            'sidewalk', 
            'parking', 
            'rail track', 
            'building',
            'wall', 
            'fence', 
            'guard rail', 
            'bridge', 
            'tunnel', 
            'pole', 
            'polegroup', 
            'traffic light',
            'traffic sign', 
            'vegetation', 
            'terrain', 
            'sky', 
            'person', 
            'rider', 
            'car', 
            'truck', 
            'bus',
            'caravan', 
            'trailer', 
            'train', 
            'motorcycle', 
            'bicycle', 
            'license plate',
        ]

        # Remap gta to new classes
        cat2id = {
            self.categories[i]: i
            for i in range(len(self.categories))
        }
        for key in self.cat_mapping:
            cat2id[key] = cat2id[self.cat_mapping[key]]
        
        self.gta_to_classid = {}
        for i in initial_classes:
            idx = initial_classes.index(i)
            self.gta_to_classid[idx+1] = cat2id[i]
        print(self.gta_to_classid)

        images_folder = os.path.join(path, 'images')
        labels_folder = os.path.join(path, 'labels')
        item_folder = os.path.join(path, 'split_list')
        item_list = os.path.join(item_folder,split + ".txt")
        items = open(item_list).read().splitlines()

        # Get images & labels paths
        for fname in sorted(os.listdir(labels_folder)):
            img_id = re.search(r'[1-9]+[0-9]*', fname).group(0)
            if img_id in items:
                img_path = os.path.join(images_folder, fname)
                label_path = os.path.join(labels_folder, fname)
                self.image_paths.append(img_path)
                self.labels_path.append(label_path)
        self.length_gta = len(self.image_paths)

    def __getitem__(self, item):
        # Load image
        image = cv2.imread(self.image_paths[item])[:, :, ::-1]
        mask = np.array(Image.open(self.labels_path[item]), dtype='int64')

        # Resize image, mask
        image = cv2.resize(image, dsize=(self.image_h, self.image_w))
        mask = cv2.resize(mask, dsize=(self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)

        for c in self.gta_to_classid:
            mask[mask == c] = self.gta_to_classid[c]

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
