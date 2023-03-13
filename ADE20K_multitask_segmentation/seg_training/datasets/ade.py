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
import copy

from PIL import Image

from . import scenenet_pb2 as sn

class ADEDataset(torch.utils.data.Dataset):
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

        self.image_names = []

        self.preproc = nnio.Preprocessing(
            dtype='float32',
            channels_first=True,
            divide_by_255=True,
        )
        
        initial_classes = [
            'background',
            'wall',
            'building;edifice',
            'sky',
            'floor;flooring',
            'tree',
            'ceiling',
            'road;route',
            'bed',
            'windowpane;window',
            'grass',
            'cabinet',
            'sidewalk;pavement',
            'person;individual;someone;somebody;mortal;soul',
            'earth;ground',
            'door;double;door',
            'table',
            'mountain;mount',
            'plant;flora;plant;life',
            'curtain;drape;drapery;mantle;pall',
            'chair',
            'car;auto;automobile;machine;motorcar',
            'water',
            'painting;picture',
            'sofa;couch;lounge',
            'shelf',
            'house',
            'sea',
            'mirror',
            'rug;carpet;carpeting',
            'field',
            'armchair',
            'seat',
            'fence;fencing',
            'desk',
            'rock;stone',
            'wardrobe;closet;press',
            'lamp',
            'bathtub;bathing;tub;bath;tub',
            'railing;rail',
            'cushion',
            'base;pedestal;stand',
            'box',
            'column;pillar',
            'signboard;sign',
            'chest;of;drawers;chest;bureau;dresser',
            'counter',
            'sand',
            'sink',
            'skyscraper',
            'fireplace;hearth;open;fireplace',
            'refrigerator;icebox',
            'grandstand;covered;stand',
            'path',
            'stairs;steps',
            'runway',
            'case;display;case;showcase;vitrine',
            'pool;table;billiard;table;snooker;table',
            'pillow',
            'screen;door;screen',
            'stairway;staircase',
            'river',
            'bridge;span',
            'bookcase',
            'blind;screen',
            'coffee;table;cocktail;table',
            'toilet;can;commode;crapper;pot;potty;stool;throne',
            'flower',
            'book',
            'hill',
            'bench',
            'countertop',
            'stove;kitchen;stove;range;kitchen;range;cooking;stove',
            'palm;palm;tree',
            'kitchen;island',
            'computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system',
            'swivel;chair',
            'boat',
            'bar',
            'arcade;machine',
            'hovel;hut;hutch;shack;shanty',
            'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle',
            'towel',
            'light;light;source',
            'truck;motortruck',
            'tower',
            'chandelier;pendant;pendent',
            'awning;sunshade;sunblind',
            'streetlight;street;lamp',
            'booth;cubicle;stall;kiosk',
            'television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box',
            'airplane;aeroplane;plane',
            'dirt;track',
            'apparel;wearing;apparel;dress;clothes',
            'pole',
            'land;ground;soil',
            'bannister;banister;balustrade;balusters;handrail',
            'escalator;moving;staircase;moving;stairway',
            'ottoman;pouf;pouffe;puff;hassock',
            'bottle',
            'buffet;counter;sideboard',
            'poster;posting;placard;notice;bill;card',
            'stage',
            'van',
            'ship',
            'fountain',
            'conveyer;belt;conveyor;belt;conveyer;conveyor;transporter',
            'canopy',
            'washer;automatic;washer;washing;machine',
            'plaything;toy',
            'swimming;pool;swimming;bath;natatorium',
            'stool',
            'barrel;cask',
            'basket;handbasket',
            'waterfall;falls',
            'tent;collapsible;shelter',
            'bag',
            'minibike;motorbike',
            'cradle',
            'oven',
            'ball',
            'food;solid;food',
            'step;stair',
            'tank;storage;tank',
            'trade;name;brand;name;brand;marque',
            'microwave;microwave;oven',
            'pot;flowerpot',
            'animal;animate;being;beast;brute;creature;fauna',
            'bicycle;bike;wheel;cycle',
            'lake',
            'dishwasher;dish;washer;dishwashing;machine',
            'screen;silver;screen;projection;screen',
            'blanket;cover',
            'sculpture',
            'hood;exhaust;hood',
            'sconce',
            'vase',
            'traffic;light;traffic;signal;stoplight',
            'tray',
            'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin',
            'fan',
            'pier;wharf;wharfage;dock',
            'crt;screen',
            'plate',
            'monitor;monitoring;device',
            'bulletin;board;notice;board',
            'shower',
            'radiator',
            'glass;drinking;glass',
            'clock',
            'flag'
        ]
        
        # Remap ade to new classes
        cat2id = {
            self.categories[i]: i
            for i in range(len(self.categories))
        }
        for key in self.cat_mapping:
            cat2id[key] = cat2id[self.cat_mapping[key]]
        
        self.ade_to_classid = {}
        for i in initial_classes:
            idx = initial_classes.index(i)
            self.ade_to_classid[idx] = cat2id[i]
        
        if self.split == 'train':
            images_folder = os.path.join(self.path, 'images/training')
            masks_folder = os.path.join(self.path, 'annotations/training')
        elif self.split == "val":
            images_folder = os.path.join(self.path, 'images/validation')
            masks_folder = os.path.join(self.path, 'annotations/validation')
        else:
            images_folder = os.path.join(self.path, 'images/validation')
            masks_folder = os.path.join(self.path, 'annotations/validation')

        for fname in sorted(os.listdir(masks_folder)):
            fname = re.split(r'.png', fname)[0]
            img_path = os.path.join(images_folder, fname + ".jpg")
            mask_path = os.path.join(masks_folder, fname + ".png")
            self.image_paths.append(img_path)
            self.masks_path.append(mask_path)
            self.image_names.append(fname)
        
        num_categories = len(self.categories) - 1
        self.ds_indices = [[] for i in range(num_categories)]
        for item, mask_path in enumerate(self.masks_path):
            mask = np.array(Image.open(mask_path).convert("P")).astype('int64')
            for c in self.ade_to_classid:
                mask[mask == c] = self.ade_to_classid[c]
            for i in range(1, num_categories + 1):
                if len(mask[mask == i]) > 0:
                    self.ds_indices[i - 1].append(item)
        print(len(self.ds_indices))

    def __getitem__(self, x):
        if isinstance(x, tuple):
            item, category_idx = x
        else:
            item = x
            category_idx = 1
        
        # Load image
        image = np.array(Image.open(self.image_paths[item]).convert('RGB'))
        mask = np.array(Image.open(self.masks_path[item]).convert("P")).astype('int64')

        image_name = self.image_names[item]
        
        # Resize image, mask
        image = cv2.resize(image, dsize=(self.image_h, self.image_w))
        mask = cv2.resize(mask, dsize=(self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)
        
        for c in self.ade_to_classid:
            mask[mask == c] = self.ade_to_classid[c]
            
        mask[mask != category_idx] *= 0
        mask[mask == category_idx] = 1

        # Augment image
        if self.augmentations is not None:
            aug = self.augmentations(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
            
        # Resize, scale, etc
        image = self.preproc(image)
        return {'image': image, 'mask': mask, 'task': category_idx, 'image_name': image_name}

    def __len__(self):
        return sum([len(x) for x in self.ds_indices]) #len(self.masks_path)
    
    def get_ds_indices(self):
        return copy.deepcopy(self.ds_indices)
