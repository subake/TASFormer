# Getting Started

This document provides a brief intro of the usage of TASFormer.

Make sure to change directory according to chosen dataset:

- For ADE20K dataset:
```bash
cd ADE20K_multitask_segmentation/
```
- For GDD, Sun, SceneNet and Trans10k datasets: 
```bash
cd multitask_segmentation/
```
## Training

- Make sure to setup wandb before training a model.

  ```bash
  wandb login
  ```
- Before first run and after making changes inside [seg_training/](ADE20K_multitask_segmentation/seg_training/) make shure to update the environment.

  ```bash
  sh setup.sh
  ```

- We provide a script [scripts/train.py](ADE20K_multitask_segmentation/scripts/train.py), that is made to train all the configs provided in TASFormer.

- To train a model, first setup the corresponding dataset config. For example, [configs/config_ade.yaml](ADE20K_multitask_segmentation/configs/config_ade.yaml).

- You can setup training hyperparameters and select model configuration. Available backbone options: `segformer`, `tasformer_with_vsa_task_embedding`, `tasformer_with_task_embedding` and `tasformer_with_adapter`.

- For `tasformer_with_adapter` you additionally have to manually update files inside `anaconda3/envs/tasformer/lib/python3.8/site-packages/transformers/models/segformer/` with files from:
  - [transformers_update_for_adapters/hf/](transformers_update_for_adapters/hf/) for TASFormer (HF adapter),
  - [transformers_update_for_adapters/hf++/](transformers_update_for_adapters/hf%2B%2B/) for TASFormer (HF adapter++).

- To train model, use:

```bash
python3 scripts/train.py --config configs/config_ade.yaml \
    --accelerator gpu --devices 1, \
    --wandb_project tasformer \
    --name "tasformer_with_adapter++_ade_150"
```

## Inference

- You can download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1at7LJZdFHpxQukhDvpf7xfctBCiFCOxg?usp=sharing) (See Table below).

- To inference a model, first setup the corresponding dataset config. For example, [configs/config_ade.yaml](ADE20K_multitask_segmentation/configs/config_ade.yaml).

- To inference model, use:

```bash
python3 scripts/inference.py --config configs/config_ade.yaml \
    --accelerator gpu --devices 1, \
    --wandb_project tasformer \
    --name "tasformer_with_adapter++_ade_150_predict" \
    --load_from ./model_tasformer_with_adapter++_ade_150.ckpt 
```

## Pretrained Models 
| Method | Params | Num Classes | Crop Size | $bIoU$, % | Checkpoint |
|   :---:| :---:   |  :---: |    :---:   |    :---:   |    :---:   |
| SegFormer (B0) | 3.8M | 2 | 320&times;320 | 65.5 | [model](https://drive.google.com/file/d/1He4BffxQ95-aGqG_mf-V7e7kOtzkpG9m/view?usp=share_link) |
| SegFormer (B0) | 3.8M | 12 | 320&times;320 | 36.6 | [model](https://drive.google.com/file/d/1l7AFDGU6CMUYpbj-lfPvuQsHsvRO43Da/view?usp=share_link) |
| SegFormer (B0) | 3.8M | 150 | 320&times;320 | 7.04 | [model](https://drive.google.com/file/d/152QlhIACRD1QJ6AwHtpnsiPTzN8RhNAe/view?usp=share_link) |
| TASFormer (emb) | 4.1M | 2 | 320&times;320 | 77.6 | [model](https://drive.google.com/file/d/1Z3HgILxH2Et0iKDwwTHfw5zMhInQlr0n/view?usp=share_link) |
| TASFormer (emb) | 4.7M | 12 | 320&times;320 | 15.1 | [model](https://drive.google.com/file/d/1OmgqJHjPIwp7T0MukHwZdEh8oyAJf77t/view?usp=share_link) |
| TASFormer (emb) | 13.6M | 150 | 320&times;320 | 28.8 | [model](https://drive.google.com/file/d/1NXYU2eGip7R3yhVVk3XZ7u2_qjLhZcu8/view?usp=share_link) |
| TASFormer (vsa emb) | 4.1M | 2 | 320&times;320 | 76.1 | [model](https://drive.google.com/file/d/1Rdl3-ANu7C7hW65MJP8QjY5mtIHOuwyS/view?usp=share_link) |
| TASFormer (vsa emb) | 4.1M | 12 | 320&times;320 | 0.59 | [model](https://drive.google.com/file/d/1DtdpiQrntwPqrxl48K7h_UP8Zwm2SMEQ/view?usp=share_link) |
| TASFormer (vsa emb) | 4.1M | 150 | 320&times;320 | 0.20 | [model](https://drive.google.com/file/d/1RwF88JoSNGKpO2c1qcvBJqw1CmN74uAR/view?usp=share_link) |
| TASFormer (HF adapter) | 7.3M | 2 | 320&times;320 | 86.5 | [model](https://drive.google.com/file/d/1K2LjAMoxjr9Kc83kF_9aUwR_vUjPLMCB/view?usp=share_link) |
| TASFormer (HF adapter) | 7.3M | 12 | 320&times;320 | 78.7 | [model](https://drive.google.com/file/d/1ALZdDXM9Mq3BWHNZmEL8aH0DsSN8QE8I/view?usp=share_link) |
| TASFormer (HF adapter) | 7.3M | 150 | 320&times;320 | 65.3 | [model](https://drive.google.com/file/d/1E8D-ahBjvp-sPonyNCPwvTWeg8iezdN5/view?usp=share_link) |
| TASFormer (HF adapter) | 7.3M |150 | 640&times;640 | 67.8 | [model](https://drive.google.com/file/d/1r8Ea7poabAPVhMJ10Is83X_yx0iNtQqv/view?usp=share_link) |
| TASFormer (HF adapter) | 7.3M | 150 | 896&times;896 | 68.9 | [model](https://drive.google.com/file/d/1gpvXgGPYZ9aPSNtRy6j34bGmfApmh3Lt/view?usp=share_link) |
| TASFormer (HF adapter++) | 5.7M | 2 | 320&times;320 | 86.3 | [model](https://drive.google.com/file/d/1Ld9Cwbc4E3yR1iTzR41LEASWK9G0qF0J/view?usp=share_link) |
| TASFormer (HF adapter++) | 5.7M | 12 | 320&times;320 | 77.2 | [model](https://drive.google.com/file/d/10RoS8hAhcjPaH5zPYoqNhLn2X_mFQIQ3/view?usp=share_link) |
| TASFormer (HF adapter++) | 5.7M | 150 | 320&times;320 | 64.0 | [model](https://drive.google.com/file/d/1zk0g2curcEJ4qwbtVYq6hpsPguLHz9Ip/view?usp=share_link) |