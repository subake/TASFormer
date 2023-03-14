# Datasets Preparation

## [ADE20K](http://sceneparsing.csail.mit.edu/)

Codes and configs for ADE20K dataset are located in `ADE20K_multitask_segmentation/`.

Please, update path to dataset inside [datasets[machine_name].yaml](https://github.com/subake/TASFormer/blob/main/ADE20K_multitask_segmentation/configs/datasets%5Bcds2%5D.yaml).

```text
# ADE20K dataset
ade_train:
  type: ade
  path: /home/data/ADE20K
  split: train
ade_val:
  type: ade
  path: /home/data/ADE20K
  split: val
```

### Expected dataset structure for ADE20K

```text
ADE20K/
  images/
    training/
    validation/
  annotations/
    training/
    validation/
```

You can check [ade.py](https://github.com/subake/TASFormer/blob/main/ADE20K_multitask_segmentation/seg_training/datasets/ade.py) to modify ADE data processing and for more details. 


## [GDD](), [Sun](), [SceneNet](), [Trans10k]()

Codes and configs for these datasets are located in `multitask_segmentation/`.

Please, update path to datasets inside [datasets[machine_name].yaml](https://github.com/subake/TASFormer/blob/main/multitask_segmentation/configs/datasets%5Bcds2%5D.yaml).

```text
# SceneNet dataset
scnet_train:
  type: scnet
  path: '/home/data/sn/train'
  protobuf_path: '/home/data/sn/train/scenenet_rgbd_train_0.pb'
  path_to_classes: '/home/data/sn/wnid_to_classes.txt'

scnet_val:
  type: scnet
  path: '/home/data/sn/val'
  protobuf_path: '/home/data/sn/val/scenenet_rgbd_val.pb'
  path_to_classes: '/home/data/sn/wnid_to_classes.txt'
```

### Expected dataset structure

You can check `dataset_name.py` inside [datasets/](https://github.com/subake/TASFormer/tree/main/multitask_segmentation/seg_training/datasets) for detailed information about dataset structure, more details and to modify data processing. 