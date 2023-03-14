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


class MetalbotDataset(torch.utils.data.Dataset):
    '''
    Loads images from folders
    '''
    def  __init__(
        self,
        path,
        image_h,
        image_w,
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


        self.preproc = nnio.Preprocessing(
            dtype='float32',
            channels_first=True,
        )

        images_folder = path
        masks_folder = None

        # Get images
        for fname in sorted(os.listdir(images_folder)):
            fname = re.split(r'.jpg', fname)[0]
            img_path = os.path.join(images_folder, fname + ".jpg")
            self.image_paths.append(img_path)

    def __getitem__(self, item):
        # Load image
        image = np.array(Image.open(self.image_paths[item]))

        # Resize image
        image = cv2.resize(image, dsize=(self.image_h, self.image_w))
        mask = np.zeros(shape=(self.image_h, self.image_w), dtype='uint8')
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