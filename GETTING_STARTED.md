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

- You can download pretrained models from [Google Drive](#pretrained-models) (See Table below).

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
| SegFormer (B0) | 3.8M | 2 | 320&times;320 | 63.2 | [model](https://drive.google.com/file/d/1PFQsBBjS0lAT41zJ3hF-KdnHmHzVQ10N/view?usp=drive_link) |
| SegFormer (B0) | 3.8M | 12 | 320&times;320 | 52.4 | [model](https://drive.google.com/file/d/1u1VhJM9tko943jBEIzrB-RSVrvQV8Qug/view?usp=drive_link) |
| SegFormer (B0) | 3.8M | 150 | 320&times;320 | 37.9 | [model](https://drive.google.com/file/d/103X3JZRk82VJlnGl4DiuyDHhRbbvrL4x/view?usp=drive_link) |
| TASFormer (emb) | 4.1M | 2 | 320&times;320 | 60.8 | [model](https://drive.google.com/file/d/1nwlKJ_NDGNaCkN1P2hqdJdGP3Fo7qqha/view?usp=drive_link) |
| TASFormer (emb) | 4.7M | 12 | 320&times;320 | 6.1 | [model](https://drive.google.com/file/d/1835-f-TKRdG6pFomqOJtFN2GptloDfQY/view?usp=drive_link) |
| TASFormer (emb) | 13.6M | 150 | 320&times;320 | 14.1 | [model](https://drive.google.com/file/d/14DbM6p6Iqm_Zqruzqo1SCp2CLZulWwWr/view?usp=drive_link) |
| TASFormer (vsa emb) | 4.1M | 2 | 320&times;320 | 48.6 | [model](https://drive.google.com/file/d/1nvukY2Jh177DxLPwJ_OTeGmPLtB87E34/view?usp=drive_link) |
| TASFormer (vsa emb) | 4.1M | 12 | 320&times;320 | 0.1 | [model](https://drive.google.com/file/d/1tTOHdSh4HlQTjZWPH9HtOB3z-7TSCy7g/view?usp=drive_link) |
| TASFormer (vsa emb) | 4.1M | 150 | 320&times;320 | 0.1 | [model](https://drive.google.com/file/d/1eDVhWDha4RLKsGfIO861MXkaeGiiOw1L/view?usp=drive_link) |
| TASFormer (HF adapter) | 7.3M | 2 | 320&times;320 | 67.9 | [model](https://drive.google.com/file/d/1XIq5DM08hGA3tv9PcvDzizrgfNLh4k0h/view?usp=drive_link) |
| TASFormer (HF adapter) | 7.3M | 12 | 320&times;320 | 59.4 | [model](https://drive.google.com/file/d/1V_0hBspI2BidHAX1VFew5yEsE6-XhZDN/view?usp=drive_link) |
| TASFormer (HF adapter) | 7.3M | 150 | 320&times;320 | 48.3 | [model](https://drive.google.com/file/d/18mkc_QybP2EMyNorj7_fSfh-eKjPj98u/view?usp=drive_link) |
| TASFormer (HF adapter) | 7.3M |150 | 640&times;640 | 51.1 | [model](https://drive.google.com/file/d/1A_A6ZC_3jYG07cLFNjLK2WYXE3ZbNA9n/view?usp=drive_link) |
| TASFormer (HF adapter) | 7.3M | 150 | 896&times;896 | 52.0 | [model](https://drive.google.com/file/d/1cCSDl0bPX9uL33rS6Q8HP0cf-LgLhjtl/view?usp=drive_link) |
| TASFormer (HF adapter++) | 5.7M | 2 | 320&times;320 | 67.9 | [model](https://drive.google.com/file/d/1uPb5AEDj2VKbjKOP3lVYPRbVhM4oced7/view?usp=drive_link) |
| TASFormer (HF adapter++) | 5.7M | 12 | 320&times;320 | 58.4 | [model](https://drive.google.com/file/d/1TGlpAisiZMY8JQ1FqMsC5UiUxA-ThBjO/view?usp=drive_link) |
| TASFormer (HF adapter++) | 5.7M | 150 | 320&times;320 | 47.6 | [model](https://drive.google.com/file/d/1SjfSbMTZBQgUhsQoz-uJd7ej6QQrgXLm/view?usp=drive_link) |
