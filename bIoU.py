
import numpy as np
import os
from PIL import Image
import cv2

# annotations
gt_path = "/home/data/ADE20K/annotations/validation"
# path to masks with all categories
pred_path = "/home/TASFormer/ADE20K_multitask_segmentation/results/adapter_150_640"

num_classes = 150
# 12
categories_ade_12=[
   "other",
   "building;edifice",
   "sky",
   "floor;flooring",
   "tree",
   "bed",
   "windowpane;window",
   "person;individual;someone;somebody;mortal;soul",
   "door;double;door",
   "table",
   "chair",
   "car;auto;automobile;machine;motorcar",
   "bicycle;bike;wheel;cycle"
]

categories_ade_2=[
   "other",
   "building;edifice",
   "floor;flooring",
]
categories_ade=[
    "other", "wall", "building;edifice", "sky", "floor;flooring", "tree", "ceiling", "road;route", "bed",
    "windowpane;window", "grass", "cabinet", "sidewalk;pavement", "person;individual;someone;somebody;mortal;soul",
    "earth;ground", "door;double;door", "table", "mountain;mount", "plant;flora;plant;life", "curtain;drape;drapery;mantle;pall",
    "chair", "car;auto;automobile;machine;motorcar", "water", "painting;picture", "sofa;couch;lounge", "shelf",
    "house","sea", "mirror", "rug;carpet;carpeting", "field", "armchair", "seat", "fence;fencing", "desk",
    "rock;stone", "wardrobe;closet;press", "lamp", "bathtub;bathing;tub;bath;tub", "railing;rail", "cushion",
    "base;pedestal;stand", "box", "column;pillar", "signboard;sign", "chest;of;drawers;chest;bureau;dresser", "counter",
    "sand", "sink", "skyscraper", "fireplace;hearth;open;fireplace", "refrigerator;icebox", "grandstand;covered;stand",
    "path", "stairs;steps", "runway", "case;display;case;showcase;vitrine", "pool;table;billiard;table;snooker;table",
    "pillow", "screen;door;screen", "stairway;staircase", "river", "bridge;span", "bookcase", "blind;screen",
    "coffee;table;cocktail;table", "toilet;can;commode;crapper;pot;potty;stool;throne", "flower", "book",
    "hill", "bench", "countertop", "stove;kitchen;stove;range;kitchen;range;cooking;stove", "palm;palm;tree",
    "kitchen;island", "computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system",
    "swivel;chair", "boat", "bar", "arcade;machine", "hovel;hut;hutch;shack;shanty", "bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle",
    "towel", "light;light;source", "truck;motortruck", "tower", "chandelier;pendant;pendent", "awning;sunshade;sunblind", "streetlight;street;lamp", "booth;cubicle;stall;kiosk", "television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box", "airplane;aeroplane;plane", "dirt;track",
    "apparel;wearing;apparel;dress;clothes", "pole", "land;ground;soil", "bannister;banister;balustrade;balusters;handrail", "escalator;moving;staircase;moving;stairway", 
    "ottoman;pouf;pouffe;puff;hassock", "bottle", "buffet;counter;sideboard", "poster;posting;placard;notice;bill;card",
    "stage", "van", "ship", "fountain", "conveyer;belt;conveyor;belt;conveyer;conveyor;transporter", "canopy",
    "washer;automatic;washer;washing;machine", "plaything;toy", "swimming;pool;swimming;bath;natatorium", "stool",
    "barrel;cask", "basket;handbasket", "waterfall;falls", "tent;collapsible;shelter", "bag", "minibike;motorbike", "cradle", 
    "oven", "ball", "food;solid;food", "step;stair", "tank;storage;tank", "trade;name;brand;name;brand;marque", 
    "microwave;microwave;oven", "pot;flowerpot", "animal;animate;being;beast;brute;creature;fauna", "bicycle;bike;wheel;cycle",
    "lake", "dishwasher;dish;washer;dishwashing;machine", "screen;silver;screen;projection;screen", "blanket;cover", 
    "sculpture", "hood;exhaust;hood", "sconce", "vase", "traffic;light;traffic;signal;stoplight", "tray", 
    "ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin", "fan",
    "pier;wharf;wharfage;dock", "crt;screen", "plate", "monitor;monitoring;device", "bulletin;board;notice;board",
    "shower", "radiator", "glass;drinking;glass", "clock", "flag"
    ]

cat_mapping_ade_12={
   'background': 'other', 
   'wall': 'other',
   'building;edifice': 'building;edifice',
   'sky': 'sky',
   'floor;flooring': 'floor;flooring',
   'tree': 'tree',
   'ceiling': 'other',
   'road;route': 'other',
   'bed': 'bed',
   'windowpane;window': 'windowpane;window',
   'grass': 'other',
   'cabinet': 'other',
   'sidewalk;pavement': 'other',
   'person;individual;someone;somebody;mortal;soul': 'person;individual;someone;somebody;mortal;soul',
   'earth;ground': 'other',
   'door;double;door': 'door;double;door',
   'table': 'table',
   'mountain;mount': 'other',
   'plant;flora;plant;life': 'other',
   'curtain;drape;drapery;mantle;pall': 'other',
   'chair': 'chair',
   'car;auto;automobile;machine;motorcar': 'car;auto;automobile;machine;motorcar',
   'water': 'other',
   'painting;picture': 'other',
   'sofa;couch;lounge': 'other',
   'shelf': 'other',
   'house': 'other',
   'sea': 'other',
   'mirror': 'other',
   'rug;carpet;carpeting': 'other',
   'field': 'other',
   'armchair': 'other',
   'seat': 'other',
   'fence;fencing': 'other',
   'desk': 'other',
   'rock;stone': 'other',
   'wardrobe;closet;press': 'other',
   'lamp': 'other',
   'bathtub;bathing;tub;bath;tub': 'other',
   'railing;rail': 'other',
   'cushion': 'other',
   'base;pedestal;stand': 'other',
   'box': 'other',
   'column;pillar': 'other',
   'signboard;sign': 'other',
   'chest;of;drawers;chest;bureau;dresser': 'other',
   'counter': 'other',
   'sand': 'other',
   'sink': 'other',
   'skyscraper': 'other',
   'fireplace;hearth;open;fireplace': 'other',
   'refrigerator;icebox': 'other',
   'grandstand;covered;stand': 'other',
   'path': 'other',
   'stairs;steps': 'other',
   'runway': 'other',
   'case;display;case;showcase;vitrine': 'other',
   'pool;table;billiard;table;snooker;table': 'other',
   'pillow': 'other',
   'screen;door;screen': 'other',
   'stairway;staircase': 'other',
   'river': 'other',
   'bridge;span': 'other',
   'bookcase': 'other',
   'blind;screen': 'other',
   'coffee;table;cocktail;table': 'other',
   'toilet;can;commode;crapper;pot;potty;stool;throne': 'other',
   'flower': 'other',
   'book': 'other',
   'hill': 'other',
   'bench': 'other',
   'countertop': 'other',
   'stove;kitchen;stove;range;kitchen;range;cooking;stove': 'other',
   'palm;palm;tree': 'other',
   'kitchen;island': 'other',
   'computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system': 'other',
   'swivel;chair': 'other',
   'boat': 'other',
   'bar': 'other',
   'arcade;machine': 'other',
   'hovel;hut;hutch;shack;shanty': 'other',
   'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle': 'other',
   'towel': 'other',
   'light;light;source': 'other',
   'truck;motortruck': 'other',
   'tower': 'other',
   'chandelier;pendant;pendent': 'other',
   'awning;sunshade;sunblind': 'other',
   'streetlight;street;lamp': 'other',
   'booth;cubicle;stall;kiosk': 'other',
   'television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box': 'other',
   'airplane;aeroplane;plane': 'other',
   'dirt;track': 'other',
   'apparel;wearing;apparel;dress;clothes': 'other',
   'pole': 'other',
   'land;ground;soil': 'other',
   'bannister;banister;balustrade;balusters;handrail': 'other',
   'escalator;moving;staircase;moving;stairway': 'other',
   'ottoman;pouf;pouffe;puff;hassock': 'other',
   'bottle': 'other',
   'buffet;counter;sideboard': 'other',
   'poster;posting;placard;notice;bill;card': 'other',
   'stage': 'other',
   'van': 'other',
   'ship': 'other',
   'fountain': 'other',
   'conveyer;belt;conveyor;belt;conveyer;conveyor;transporter': 'other',
   'canopy': 'other',
   'washer;automatic;washer;washing;machine': 'other',
   'plaything;toy': 'other',
   'swimming;pool;swimming;bath;natatorium': 'other',
   'stool': 'other',
   'barrel;cask': 'other',
   'basket;handbasket': 'other',
   'waterfall;falls': 'other',
   'tent;collapsible;shelter': 'other',
   'bag': 'other',
   'minibike;motorbike': 'other',
   'cradle': 'other',
   'oven': 'other',
   'ball': 'other',
   'food;solid;food': 'other',
   'step;stair': 'other',
   'tank;storage;tank': 'other',
   'trade;name;brand;name;brand;marque': 'other',
   'microwave;microwave;oven': 'other',
   'pot;flowerpot': 'other',
   'animal;animate;being;beast;brute;creature;fauna': 'other',
   'bicycle;bike;wheel;cycle': 'bicycle;bike;wheel;cycle',
   'lake': 'other',
   'dishwasher;dish;washer;dishwashing;machine': 'other',
   'screen;silver;screen;projection;screen': 'other',
   'blanket;cover': 'other',
   'sculpture': 'other',
   'hood;exhaust;hood': 'other',
   'sconce': 'other',
   'vase': 'other',
   'traffic;light;traffic;signal;stoplight': 'other',
   'tray': 'other',
   'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin': 'other',
   'fan': 'other',
   'pier;wharf;wharfage;dock': 'other',
   'crt;screen': 'other',
   'plate': 'other',
   'monitor;monitoring;device': 'other',
   'bulletin;board;notice;board': 'other',
   'shower': 'other',
   'radiator': 'other',
   'glass;drinking;glass': 'other',
   'clock': 'other',
   'flag': 'other',
}
cat_mapping_ade_2={
   'background': 'other', 
   'wall': 'other',
   'building;edifice': 'building;edifice',
   'sky': 'other',
   'floor;flooring': 'floor;flooring',
   'tree': 'other',
   'ceiling': 'other',
   'road;route': 'other',
   'bed': 'other',
   'windowpane;window': 'other',
   'grass': 'other',
   'cabinet': 'other',
   'sidewalk;pavement': 'other',
   'person;individual;someone;somebody;mortal;soul': 'other',
   'earth;ground': 'other',
   'door;double;door': 'other',
   'table': 'other',
   'mountain;mount': 'other',
   'plant;flora;plant;life': 'other',
   'curtain;drape;drapery;mantle;pall': 'other',
   'chair': 'other',
   'car;auto;automobile;machine;motorcar': 'other',
   'water': 'other',
   'painting;picture': 'other',
   'sofa;couch;lounge': 'other',
   'shelf': 'other',
   'house': 'other',
   'sea': 'other',
   'mirror': 'other',
   'rug;carpet;carpeting': 'other',
   'field': 'other',
   'armchair': 'other',
   'seat': 'other',
   'fence;fencing': 'other',
   'desk': 'other',
   'rock;stone': 'other',
   'wardrobe;closet;press': 'other',
   'lamp': 'other',
   'bathtub;bathing;tub;bath;tub': 'other',
   'railing;rail': 'other',
   'cushion': 'other',
   'base;pedestal;stand': 'other',
   'box': 'other',
   'column;pillar': 'other',
   'signboard;sign': 'other',
   'chest;of;drawers;chest;bureau;dresser': 'other',
   'counter': 'other',
   'sand': 'other',
   'sink': 'other',
   'skyscraper': 'other',
   'fireplace;hearth;open;fireplace': 'other',
   'refrigerator;icebox': 'other',
   'grandstand;covered;stand': 'other',
   'path': 'other',
   'stairs;steps': 'other',
   'runway': 'other',
   'case;display;case;showcase;vitrine': 'other',
   'pool;table;billiard;table;snooker;table': 'other',
   'pillow': 'other',
   'screen;door;screen': 'other',
   'stairway;staircase': 'other',
   'river': 'other',
   'bridge;span': 'other',
   'bookcase': 'other',
   'blind;screen': 'other',
   'coffee;table;cocktail;table': 'other',
   'toilet;can;commode;crapper;pot;potty;stool;throne': 'other',
   'flower': 'other',
   'book': 'other',
   'hill': 'other',
   'bench': 'other',
   'countertop': 'other',
   'stove;kitchen;stove;range;kitchen;range;cooking;stove': 'other',
   'palm;palm;tree': 'other',
   'kitchen;island': 'other',
   'computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system': 'other',
   'swivel;chair': 'other',
   'boat': 'other',
   'bar': 'other',
   'arcade;machine': 'other',
   'hovel;hut;hutch;shack;shanty': 'other',
   'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle': 'other',
   'towel': 'other',
   'light;light;source': 'other',
   'truck;motortruck': 'other',
   'tower': 'other',
   'chandelier;pendant;pendent': 'other',
   'awning;sunshade;sunblind': 'other',
   'streetlight;street;lamp': 'other',
   'booth;cubicle;stall;kiosk': 'other',
   'television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box': 'other',
   'airplane;aeroplane;plane': 'other',
   'dirt;track': 'other',
   'apparel;wearing;apparel;dress;clothes': 'other',
   'pole': 'other',
   'land;ground;soil': 'other',
   'bannister;banister;balustrade;balusters;handrail': 'other',
   'escalator;moving;staircase;moving;stairway': 'other',
   'ottoman;pouf;pouffe;puff;hassock': 'other',
   'bottle': 'other',
   'buffet;counter;sideboard': 'other',
   'poster;posting;placard;notice;bill;card': 'other',
   'stage': 'other',
   'van': 'other',
   'ship': 'other',
   'fountain': 'other',
   'conveyer;belt;conveyor;belt;conveyer;conveyor;transporter': 'other',
   'canopy': 'other',
   'washer;automatic;washer;washing;machine': 'other',
   'plaything;toy': 'other',
   'swimming;pool;swimming;bath;natatorium': 'other',
   'stool': 'other',
   'barrel;cask': 'other',
   'basket;handbasket': 'other',
   'waterfall;falls': 'other',
   'tent;collapsible;shelter': 'other',
   'bag': 'other',
   'minibike;motorbike': 'other',
   'cradle': 'other',
   'oven': 'other',
   'ball': 'other',
   'food;solid;food': 'other',
   'step;stair': 'other',
   'tank;storage;tank': 'other',
   'trade;name;brand;name;brand;marque': 'other',
   'microwave;microwave;oven': 'other',
   'pot;flowerpot': 'other',
   'animal;animate;being;beast;brute;creature;fauna': 'other',
   'bicycle;bike;wheel;cycle': 'other',
   'lake': 'other',
   'dishwasher;dish;washer;dishwashing;machine': 'other',
   'screen;silver;screen;projection;screen': 'other',
   'blanket;cover': 'other',
   'sculpture': 'other',
   'hood;exhaust;hood': 'other',
   'sconce': 'other',
   'vase': 'other',
   'traffic;light;traffic;signal;stoplight': 'other',
   'tray': 'other',
   'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin': 'other',
   'fan': 'other',
   'pier;wharf;wharfage;dock': 'other',
   'crt;screen': 'other',
   'plate': 'other',
   'monitor;monitoring;device': 'other',
   'bulletin;board;notice;board': 'other',
   'shower': 'other',
   'radiator': 'other',
   'glass;drinking;glass': 'other',
   'clock': 'other',
   'flag': 'other',
}

iou_scores_gt = []
for idx, gt_filename in enumerate(os.listdir(gt_path)):
    #gt_filename = "ADE_val_00000001.png"
    gt_mask = np.array(Image.open(f"{gt_path}/{gt_filename}"))
    gt_mask = gt_mask.astype(np.int64)
    gt_idx = gt_filename.split(".")[0]
    for cat in range(1, 150+1):

        gt = gt_mask == cat
        if num_classes == 12:
            gt_cat_name = categories_ade[cat]
            pred_cat_name = cat_mapping_ade_12[gt_cat_name]
            pred_cat_idx = categories_ade_12.index(pred_cat_name)-1
        if num_classes == 2:
            gt_cat_name = categories_ade[cat]
            pred_cat_name = cat_mapping_ade_2[gt_cat_name]
            pred_cat_idx = categories_ade_2.index(pred_cat_name)-1
        if num_classes == 150:
            pred_cat_idx = cat-1

        if np.sum(gt) > 0 and pred_cat_idx >= 0:
            pred = np.array(Image.open(f"{pred_path}/{gt_idx}_{pred_cat_idx}.png"))
            pred = cv2.resize(pred, dsize=(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred = pred.astype(bool)

            intersection = np.logical_and(gt, pred)
            union = np.logical_or(gt, pred)
            iou_score = np.sum(intersection) / np.sum(union)
            
            iou_scores_gt.append(iou_score)
    #input()

    print("Done: ", idx)
    print("Mean IoU GT:", np.mean(iou_scores_gt))
print("Mean IoU GT:", np.mean(iou_scores_gt))