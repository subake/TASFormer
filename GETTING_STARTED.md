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
- Before first run and after making changes inside `seg_training/` make shure to update the environment.

  ```bash
  python3 setup.py bdist_wheel; pip3 install --force-reinstall --no-deps dist/*.whl
  ```

- We provide a script `scripts/train.py`, that is made to train all the configs provided in OneFormer.

- To train a model, first setup the corresponding dataset config. For example, [configs/config_ade.yaml](https://github.com/subake/TASFormer/blob/main/ADE20K_multitask_segmentation/configs/config_ade.yaml).

- You can setup training hyperparameters and select model configuration. Available options: `segformer`, `segformer_with_vsa_task_embedding`, `segformer_with_task_embedding` and `segformer_with_adapter`.

- For `segformer_with_adapter` you additionally have to manually update files inside `anaconda3/envs/tasformer/lib/python3.8/site-packages/transformers/models/segformer/` with files from:
  - [transformers_update_for_adapters/hf/](https://github.com/subake/TASFormer/tree/main/transformers_update_for_adapters/hf) for TASFormer (HF adapter)
  - [transformers_update_for_adapters/hf++/](https://github.com/subake/TASFormer/tree/main/transformers_update_for_adapters/hf%2B%2B) for TASFormer (HF adapter++)

- To train model, use:

```bash
python3 scripts/train.py --config configs/config_ade.yaml \
    --accelerator gpu --devices 1, \
    --wandb_project tasformer \
    --name "segformer_with_adapter++_ade_150"
```

## Inference

- You can download pretrained models from [Google Drive]() (See Table below).

- To inference a model, first setup the corresponding dataset config. For example, [configs/config_ade.yaml](https://github.com/subake/TASFormer/blob/main/ADE20K_multitask_segmentation/configs/config_ade.yaml).

- To inference model, use:

```bash
python3 scripts/inference.py --config configs/config_ade.yaml \
    --accelerator gpu --devices 1, \
    --wandb_project tasformer \
    --name "segformer_with_adapter++_ade_150_predict" \
    --load_from ./model_segformer_with_adapter++_ade_150.ckpt 
```

## Pretrained Models 
| Method | Params | Num Classes | Crop Size | $bIoU$, % | Checkpoint |
|   :---:| :---:   |  :---: |    :---:   |    :---:   |    :---:   |
| TASFormer (HF adapter) | 7.3M | 2 | 320&times;320 | 86.5 | [model]() |
| TASFormer (HF adapter) | 7.3M | 12 | 320&times;320 | 78.7 | [model]() |
| TASFormer (HF adapter) | 7.3M | 150 | 320&times;320 | 65.3 | [model]() |
| TASFormer (HF adapter) | 7.3M |150 | 640&times;640 | 67.8 | [model]() |
| TASFormer (HF adapter) | 7.3M | 150 | 896&times;896 | 68.9 | [model]() |
| TASFormer (HF adapter++) | 5.7M | 2 | 320&times;320 | 86.3 | [model]() |
| TASFormer (HF adapter++) | 5.7M | 12 | 320&times;320 | 77.2 | [model]() |
| TASFormer (HF adapter++) | 5.7M | 150 | 320&times;320 | 64.0 | [model]() |