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

class CityScapesDataset(torch.utils.data.Dataset):
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

        self.preproc = nnio.Preprocessing(
            dtype='float32',
            channels_first=True,
            divide_by_255=True,
        )

        initial_classes = [
          'unlabeled',
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

        # Remap cityscapes to new classes
        cat2id = {
            self.categories[i]: i
            for i in range(len(self.categories))
        }
        for key in self.cat_mapping:
            cat2id[key] = cat2id[self.cat_mapping[key]]
        
        self.city_to_classid = {}
        for i in initial_classes:
            idx = initial_classes.index(i)
            self.city_to_classid[idx] = cat2id[i]

        self.images, self.mask_paths = self._get_city_pairs()
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[index]))
        
        img = cv2.resize(img, dsize=(self.image_h, self.image_w))
        mask = cv2.resize(mask, dsize=(self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype('uint8')

        for c in self.city_to_classid:
            mask[mask == c] = self.city_to_classid[c]

        if self.augmentations is not None:
            aug = self.augmentations(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
        img = self.preproc(img)
        return {'image': img, 'mask': mask}

    def __len__(self):
        return len(self.images)

    def _get_city_pairs(self):
        def get_path_pairs(img_folder, mask_folder):
            img_paths = []
            mask_paths = []
            for root, _, files in os.walk(img_folder):
                for filename in files:
                    if filename.startswith('._'):
                        continue
                    if filename.endswith('.png'):
                        imgpath = os.path.join(root, filename)
                        foldername = os.path.basename(os.path.dirname(imgpath))
                        maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                        maskpath = os.path.join(mask_folder, foldername, maskname)
                        if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                            img_paths.append(imgpath)
                            mask_paths.append(maskpath)
                        else:
                            raise RuntimeError('cannot find the mask or image')
            return img_paths, mask_paths

        if self.split in ('train', 'val'):
            img_folder = os.path.join(self.path, 'leftImg8bit/' + self.split)
            mask_folder = os.path.join(self.path, 'gtFine/' + self.split)
            img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
            return img_paths, mask_paths
        else:
            train_img_folder = os.path.join(self.path, 'leftImg8bit/train')
            train_mask_folder = os.path.join(self.path, 'gtFine/train')
            val_img_folder = os.path.join(self.path, 'leftImg8bit/val')
            val_mask_folder = os.path.join(self.path, 'gtFine/val')
            train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
            val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
            img_paths = train_img_paths + val_img_paths
            mask_paths = train_mask_paths + val_mask_paths
        return img_paths, mask_paths