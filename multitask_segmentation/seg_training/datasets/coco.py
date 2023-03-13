import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from typing import Optional
import torch
from torch.utils.data import DataLoader

from pycocotools.coco import COCO

import nnio
import re
import json
import sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class CocoDataset(torch.utils.data.Dataset):
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

        self.categories = categories

        self.cat_mapping = cat_mapping

        self.path = path

        self.image_h = image_h
        
        self.image_w = image_w

        self.image_paths = []

        self.fnames = []

        self.preproc = nnio.Preprocessing(
            dtype='float32',
            channels_first=True,
            divide_by_255=True,
        )
        self.initial_classes = [
            'person',
            'bicycle',
            'car',
            'motorcycle',
            'airplane',
            'bus',
            'train',
            'truck',
            'boat',
            'traffic light',
            'fire hydrant',
            'stop sign',
            'parking meter',
            'bench',
            'bird',
            'cat',
            'dog',
            'horse',
            'sheep',
            'cow',
            'elephant',
            'bear',
            'zebra',
            'giraffe',
            'backpack',
            'umbrella',
            'handbag',
            'tie',
            'suitcase',
            'frisbee',
            'skis',
            'snowboard',
            'sports ball',
            'kite',
            'baseball bat',
            'baseball glove',
            'skateboard',
            'surfboard',
            'tennis racket',
            'bottle',
            'wine glass',
            'cup',
            'fork',
            'knife',
            'spoon',
            'bowl',
            'banana',
            'apple',
            'sandwich',
            'orange',
            'broccoli',
            'carrot',
            'hot dog',
            'pizza',
            'donut',
            'cake',
            'chair',
            'couch',
            'potted plant',
            'bed',
            'dining table',
            'toilet',
            'tv',
            'laptop',
            'mouse',
            'remote',
            'keyboard',
            'cell phone',
            'microwave',
            'oven',
            'toaster',
            'sink',
            'refrigerator',
            'book',
            'clock',
            'vase',
            'scissors',
            'teddy bear',
            'hair drier',
            'toothbrush',
        ]

        # Remap coco to new classes
        cat2id = {
            self.categories[i]: i
            for i in range(len(self.categories))
        }
        for key in self.cat_mapping:
            cat2id[key] = cat2id[self.cat_mapping[key]]
        
        self.coco_to_classid = {}
        for i in self.initial_classes:
            idx = self.initial_classes.index(i)
            self.coco_to_classid[idx+1] = cat2id[i]
        print(self.coco_to_classid)

        images_folder = os.path.join(path, 'IMAGES')
        annot_folder = os.path.join(self.path, 'ANNOTATIONS')
        annFile_path = os.path.join(annot_folder, 'instances.json')

        annFile = annFile_path.format(images_folder, 'IMAGES')
        with HiddenPrints():
            self.coco = COCO(annFile)

        img_ann = self.parse_json()

        # Get images paths
        for fname in sorted(os.listdir(images_folder)):
            img_id = re.search(r'[1-9]+[0-9]*', fname).group(0)
            if fname in img_ann:
                img_path = os.path.join(images_folder, fname)
                self.image_paths.append(img_path)
                self.fnames.append(fname)
        self.length_coco = len(self.image_paths)

    def __getitem__(self, item):
        # Load image
        image = cv2.imread(self.image_paths[item])[:, :, ::-1]
        mask = self.mask_for_img(item, image.shape)

        # Resize image, mask
        image = cv2.resize(image, dsize=(self.image_h, self.image_w))
        mask = cv2.resize(mask, dsize=(self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)

        for c in self.coco_to_classid:
            mask[mask == c] = self.coco_to_classid[c]

        # Augment image
        if self.augmentations is not None:
            aug = self.augmentations(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        # Resize, scale, etc
        image = self.preproc(image)
        mask = mask
        return {'image': image, 'mask': mask}

    def __len__(self):
        return len(self.image_paths)

    def mask_for_img(self, item, image):  

        annot_folder = os.path.join(self.path, 'ANNOTATIONS')
        annFile_path = os.path.join(annot_folder, 'instances.json')
        fname = self.fnames[item]

        # Get image id and annotation
        ID = re.search(r'[1-9]+[0-9]*', fname).group(0)
        catIds = self.coco.getCatIds(catNms=self.initial_classes)

        annIds = self.coco.getAnnIds(imgIds=int(ID), catIds=catIds, iscrowd=0)
        anns = self.coco.loadAnns(annIds)

        # Get masks (pixel value == category_id)
        mask_class = np.zeros((image[0], image[1]) , dtype='int64')
        if len(anns)>0:
            for i in range(len(anns)):
                new_mask = self.coco.annToMask(anns[i]) * (catIds.index(anns[i]['category_id']))
                mask_class = np.maximum(mask_class, new_mask)
        return (mask_class)

    def parse_json(self):
        annot_folder = os.path.join(self.path, 'ANNOTATIONS')
        annFile_path = os.path.join(annot_folder, 'instances.json')

        with open(annFile_path) as json_file:
            self.ann_json = json.load(json_file)

        img_ann = {img['file_name'] for img in self.ann_json['images']}
        return(img_ann)
