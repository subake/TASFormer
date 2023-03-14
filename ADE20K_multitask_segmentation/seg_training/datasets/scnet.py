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

class SceneNetDataset(torch.utils.data.Dataset):
    '''
    Loads images from folders
    '''

    def __init__(
            self,
            path,
            image_h,
            image_w,
            protobuf_path,
            path_to_classes,
            categories,
            cat_mapping,
            augmentations=None,
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

        self.protobuf_path = protobuf_path

        self.path_to_classes = path_to_classes

        self.categories = categories

        self.cat_mapping = cat_mapping

        NYU_13_CLASSES = [
            'Unknown', 
            'Bed', 
            'Books', 
            'Ceiling', 
            'Chair',
            'Floor', 
            'Furniture', 
            'Objects', 
            'Picture',
            'Sofa', 
            'Table', 
            'TV', 
            'Wall', 
            'Window'
        ]
        
        # Remap wnid to classes
        cat2id = {
            self.categories[i]: i
            for i in range(len(self.categories))
        }
        for key in self.cat_mapping:
            cat2id[key] = cat2id[self.cat_mapping[key]]

        column_name = '13_classes'
        class_list = NYU_13_CLASSES

        wnid_to_classid = {}
        with open(self.path_to_classes,'r') as f:
            class_lines = f.readlines()
            column_headings = class_lines[0].split()
            for class_line in class_lines[1:]:
                wnid = class_line.split()[0].zfill(8)
                classid = cat2id[class_line.split()[column_headings.index(column_name)]]
                wnid_to_classid[wnid] = classid
                if wnid_to_classid[wnid] == '3773035':
                    wnid_to_classid[wnid] = 4

        self.NYU_WNID_TO_CLASS = wnid_to_classid

        self.preproc = nnio.Preprocessing(
            dtype='float32',
            channels_first=True,
            divide_by_255=True,
        )

        trajectories = sn.Trajectories()
        try:
            with open(protobuf_path, 'rb') as f:
                trajectories.ParseFromString(f.read())
        except IOError:
            print('Scenenet protobuf data not found at location:{0}'.format(path))
            print('Please ensure you have copied the pb file to the data directory')

        self.all_views = []
        for traj in trajectories.trajectories:
            if self.augmentations is not None:
                for view in traj.views:
                    self.all_views.append((traj, view))
            else:
                view = traj.views[0]
                self.all_views.append((traj, view))

    def __getitem__(self, item):
        instance_class_map = {}
        traj, view = self.all_views[item]
        for instance in traj.instances:
            instance_type = sn.Instance.InstanceType.Name(instance.instance_type)
            if instance.instance_type != sn.Instance.BACKGROUND:
                instance_class_map[instance.instance_id] = self.NYU_WNID_TO_CLASS[instance.semantic_wordnet_id]

        instance_path = self.instance_path_from_view(traj.render_path, view)
        photo_path = self.photo_path_from_view(traj.render_path, view)
        mask = self.class_from_instance(instance_path, instance_class_map)
        image = np.array(Image.open(photo_path)).astype('uint8')

        image = cv2.resize(image, dsize=(self.image_h, self.image_w))
        mask = cv2.resize(mask, dsize=(self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype('uint8')

        if self.augmentations is not None:
            aug = self.augmentations(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        image = self.preproc(image)
        return {'image': image, 'mask': mask}

    def __len__(self):
        return len(self.all_views)

    def instance_path_from_view(self, render_path, view):
        instance_path = os.path.join(render_path, 'instance')
        image_path = os.path.join(instance_path, '{0}.png'.format(view.frame_num))
        return os.path.join(self.path, image_path)

    def photo_path_from_view(self, render_path, view):
        photo_path = os.path.join(render_path, 'photo')
        image_path = os.path.join(photo_path, '{0}.jpg'.format(view.frame_num))
        return os.path.join(self.path, image_path)

    def class_from_instance(self, instance_path, instance_class_map):
        instance_img = np.asarray(Image.open(instance_path))
        class_img = np.zeros(instance_img.shape)

        for instance, semantic_class in instance_class_map.items():
            class_img[instance_img == instance] = semantic_class
        return class_img
