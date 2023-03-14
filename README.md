# TASFormer: Task-aware Image Segmentation Transformer

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) ![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)



[[`arXiv`]()] [[`pdf`]()] [[`BibTeX`]()]

This repo contains the code for our paper **TASFormer: Task-aware Image Segmentation Transformer**.

## Contents

2. [Notes](#notes)
3. [Installation Instructions](#installation-instructions)
4. [Dataset Preparation](#dataset-preparation)
5. [Execution Instructions](#execution-instructions)
    - [Training](#training)
    - [Evaluation](#evaluation)
6. [Results](#results)

## Notes

[multitask_segmentation/](multitask_segmentation/) contains codes and configs for GDD, Sun, SceneNet and Trans10k datasets.

[ADE20K_multitask_segmentation/](ADE20K_multitask_segmentation/) contains codes and configs for ADE20K dataset.

[transformers_update_for_adapters/](transformers_update_for_adapters/) contains files required for `TASFormer (HF adapter)` and `TASFormer (HF adapter++)`.

You can find $bIoU$ metric implementation inside `validation_step()` in [pl_module.py](ADE20K_multitask_segmentation/seg_training/pl_module.py). Keep in mind, our TASFormer model works with binary masks, and averaging is performed over all masks, regardless of their class. More details about $bIoU$ metric can be found in our [Paper](). 

## Installation Instructions

- We use Python 3.8, PyTorch 1.8.1 (CUDA 10.1 build).
- We use PyTorch Lightning 1.5.0.
- For complete installation instructions, please see [Installation](INSTALL.md).

## Dataset Preparation

- We experiment on ADE20K benchmark and other datasets.
- Please see [Preparing Datasets](DATASET_PREPARATION.md) for complete instructions for preparing the datasets.

## Execution Instructions

### Training

- Please see [Getting Started](GETTING_STARTED.md) for training commands.

### Evaluation

- Please see [Getting Started](GETTING_STARTED.md) for evaluation commands.

## Results

You can find our pretrained models in [Getting Started](GETTING_STARTED.md).

![Results](images/TASFormer_visualization.png)

### Glass and Floor Segmentation

Results for glass and floor segmentation, $bIoU$, %.

| Method | Params | Crop Size | SUN RGB-D | GDD | Trans10k |
|   :---:|  :---:           | :---:               | :---:   |  :---: |    :---:   |
| SegFormer (B0) | 3.8M | 320&times;320 | 83.1 | 82.5 | 88.4 |
| TASFormer (emb) | 4.1M | 320&times;320 | 82.6 | 82.4 | 90.1 |
| TASFormer (VSA emb) | 4.1M | 320&times;320 | 81.0 | 79.7 | 90.2 |
| TASFormer (HF adapter) | 7.3M | 320&times;320 | **83.8** | **87.4** | **92.5** |

### ADE20K

Segmentation results on different `Num Classes` from ADE20K dataset, $bIoU$, %.

| Method | Params | Crop Size | 2 | 12 | 150 |
|   :---:|  :---:           | :---:               | :---:   |  :---: |    :---:   |
| SegFormer (B0) | 3.8M | 320&times;320 | 65.5 | 36.6 | 7.04 |
| TASFormer (emb) | 4--14M | 320&times;320 | 77.6 | 15.1 | 28.8 |
| TASFormer (VSA emb) | 4.1M | 320&times;320 | 76.1 | 0.59 | 0.20 |
| TASFormer (HF adapter) | 7.3M | 320&times;320 | **86.5** | **78.7** | **65.3** |
| TASFormer (HF adapter++) | 5.7M | 320&times;320 | 86.3 | 77.2 | 64.0 |

<br/><br/>

Segmentation results of TASFormer (HF adapter) with different `Crop Size` on ADE20K dataset, $bIoU$, %.

| Method | Params | Crop Size | $bIoU$, % |
|   :---:| :---:   |  :---: |    :---:   |
| TASFormer (HF adapter) | 7.3M | 320&times;320 | 65.3 |
| TASFormer (HF adapter) | 7.3M | 640&times;640 | 67.8 |
| TASFormer (HF adapter) | 7.3M | 896&times;896 | **68.9** |
