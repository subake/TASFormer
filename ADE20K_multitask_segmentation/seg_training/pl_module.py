'''
Model class using pytorch lightning
'''

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from torchmetrics import IoU, JaccardIndex

from . import networks
from .utils import visualization

import time

import matplotlib.pyplot as plt
import cv2

class LitSeg(pl.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.num_categories is None:
            num_output_channels = len(self.hparams.categories)
        else:
            num_output_channels = self.hparams.num_categories

        if num_output_channels == 2:
            num_output_channels = 1
        self.num_output_channels = num_output_channels

        self.dataset_names = {
            'train': self.hparams.train_datasets,
            'valid': self.hparams.dev_datasets,
            'test': self.hparams.test_datasets
        }

        if self.hparams.backbone == 'tasformer_with_task_embedding':
            self.mask_net = networks.tasformer_with_task_embedding.TASFormerEmbedding(
                num_classes=num_output_channels,
                num_tasks=self.hparams.num_tasks,
            )
        elif self.hparams.backbone == 'tasformer_with_vsa_task_embedding':
            self.mask_net = networks.tasformer_with_vsa_task_embedding.TASFormerVSAEmbedding(
                num_classes=num_output_channels,
            )
        elif self.hparams.backbone == 'tasformer_with_adapter':
            self.mask_net = networks.tasformer_with_adapter.TASFormerAdapter(
                num_classes=num_output_channels,
            )
        elif self.hparams.backbone == 'segformer':
            self.mask_net = networks.segformer.SegFormer(
                num_classes=num_output_channels,
            ) 
        else:
            raise BaseException('backbone must be [segformer, tasformer_with_task_embedding, tasformer_with_vsa_task_embedding, tasformer_with_adapter]')

        # Losses
        if num_output_channels <= 2:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model Parameters")
        parser.add_argument('--backbone', type=str, default=None)
        parser.add_argument('--image_h', type=int, default=None)
        parser.add_argument('--image_w', type=int, default=None)
        parser.add_argument('--num_categories', type=int, default=None)
        parser.add_argument('--dropout', type=float, default=None)
        parser.add_argument('--learning_rate', type=float, default=None)
        parser.add_argument('--log_plot_freq', type=int, default=None)
        parser.add_argument('--num_tasks', type=int, default=None)
        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def forward(self, image):
        cat_num = self.num_output_channels
        if cat_num == 1:
            cat_num += 1
        if self.hparams.backbone == 'tasformer_with_task_embedding' \
            or self.hparams.backbone == 'tasformer_with_vsa_task_embedding' \
            or self.hparams.backbone == 'tasformer_with_adapter':
            outp1 = self.mask_net(image, 0)
            
            mask_logits = [outp1['out'][0]]
        else:
            outp = self.mask_net(image)
            mask_logits = outp['out']

        if cat_num <= 2:
            mask_preds = [(mask_logits[i][0, 0] > 0).to(torch.int64) for i in range(len(mask_logits))]
        else:
            mask_preds = [torch.argmax(mask_logits[i][0], dim=0).to(torch.int64) for i in range(len(mask_logits))]

        mask = mask_preds[0]

        return mask

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        cat_num = self.num_output_channels
        if cat_num == 1:
            cat_num += 1 
                    
        for cur_task in range(1, self.hparams.num_tasks + 1):
            # Predict mask
            batch_mask = torch.clone(batch['mask'])
            batch_mask[batch_mask != cur_task] *= 0
            batch_mask[batch_mask == cur_task] = 1

            with torch.no_grad():
                if self.hparams.backbone == 'tasformer_with_task_embedding' \
                    or self.hparams.backbone == 'tasformer_with_vsa_task_embedding' \
                    or self.hparams.backbone == 'tasformer_with_adapter':
                    
                    outp = self.mask_net(torch.clone(batch['image']), cur_task - 1)
                    mask_logits = outp['out'][0]
                else:
                    outp = self.mask_net(batch['image'])
                    mask_logits = outp['out'][cur_task]
            
            if cat_num <= 2:
                mask_preds = (mask_logits[0, 0]>0).detach().cpu().numpy().astype('uint8')
            else:
                mask_preds = [torch.argmax(mask_logits[i][0], dim=0).detach().cpu().numpy().astype('uint8') for i in range(len(mask_logits))]

            cv2.imwrite(f'{self.hparams.predict_save_path}{batch["image_name"][0]}_{cur_task - 1}.png', mask_preds)

        return mask_preds
 
    def training_step(self, batch, batch_idx, mode='train'):
        cat_num = self.num_output_channels
        loss = 0

        for i, cur_batch in enumerate(batch):
            cur_task = cur_batch['task'][0]
            if self.hparams.backbone == 'tasformer_with_task_embedding' \
                or self.hparams.backbone == 'tasformer_with_vsa_task_embedding' \
                or self.hparams.backbone == 'tasformer_with_adapter':
                outp = self.mask_net(cur_batch['image'], cur_task - 1)
                mask_logits = outp['out'][0]
            else:
                outp = self.mask_net(cur_batch['image'])
                mask_logits = outp['out'][cur_task]

            if cat_num <= 2:
                cur_loss = self.loss_fn(mask_logits[:, 0], cur_batch['mask'].to(torch.float32))
            else:
                cur_loss = self.loss_fn(mask_logits[cur_task], cur_batch['mask'].long())
            loss += cur_loss

            self.log(f'{self.dataset_names[mode][i]}/loss', cur_loss)
        
        if loss != loss:
            print('ERROR: NaN loss!')

        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0, mode='valid'):
        cat_num = self.num_output_channels
        if cat_num == 1:
            cat_num += 1
        
        cur_task = batch['task'][0]

        # Predict mask
        with torch.no_grad():
            if self.hparams.backbone == 'tasformer_with_task_embedding' \
                or self.hparams.backbone == 'tasformer_with_vsa_task_embedding' \
                or self.hparams.backbone == 'tasformer_with_adapter':
                outp = self.mask_net(batch['image'], cur_task - 1)
                mask_logits = outp['out'][0]
            else:
                outp = self.mask_net(batch['image'])
                mask_logits = outp['out'][cur_task]

            mask = mask_logits
        
        if cat_num <= 2:
            # Compute loss
            loss = self.loss_fn(mask_logits[:, 0], batch['mask'].to(torch.float32))

            # Compute IoU metric
            mask_iou = (mask_logits[:, 0] > 0).detach().cpu().to(torch.int64)
            batch_iou = batch['mask'].detach().cpu().to(torch.int32)
        elif cat_num > 2:
            # Compute loss
            loss = self.loss_fn(mask_logits[cur_task], batch['mask'].long())

            # Compute IoU metric
            mask_iou = torch.argmax(mask_logits[cur_task], dim=1).detach().cpu().to(torch.int64)
            batch_iou = batch['mask'].detach().cpu().to(torch.int64)

        iou = IoU(cat_num, ignore_index=0)
        iou = iou(mask_iou, batch_iou)

        self.log(f'{self.dataset_names[mode][dataloader_idx]}/iou', iou, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f'{self.dataset_names[mode][dataloader_idx]}/loss', loss, on_step=False, on_epoch=True, add_dataloader_idx=False)
        
        self.log(f'{self.dataset_names[mode][dataloader_idx]}/{cur_task}/iou', iou, on_step=False, on_epoch=True, add_dataloader_idx=False)

        if batch_idx % self.hparams.log_plot_freq == 0:
            img = batch['image'][0].detach().cpu().numpy().transpose(1, 2, 0)
            if cat_num <= 2:
                mask_preds = (mask_logits[0, 0]>0).detach().cpu().numpy().astype('uint8')
            else:
                mask_preds = [torch.argmax(mask_logits[i][0], dim=0).detach().cpu().numpy().astype('uint8') for i in range(len(mask_logits))]

            mask_gt = batch['mask'][0].detach().cpu().numpy().astype('uint8')

            # Set colormap
            np.random.seed(1)
            colormap = np.random.random([cat_num, 3])
            colormap[0, :] = 0

            seg_mask_preds = np.zeros(list(mask_preds.shape) + [3])
            seg_mask_gt = np.zeros(list(mask_gt.shape) + [3])

            for i in range(cat_num):
                seg_mask_preds[mask_preds == i] = colormap[i]
                seg_mask_gt[mask_gt == i] = colormap[i]

            # Plot image and mask
            fig, axes = plt.subplots(4, 1, figsize=(6, 12))
            axes[0].set_title(f'Image {cur_task}')
            axes[0].imshow(img, aspect='auto')
            axes[0].axis('off')

            axes[1].set_title('Predicted mask')
            axes[1].imshow(seg_mask_preds, aspect='auto')

            axes[2].set_title('Ground truth mask')
            axes[2].imshow(seg_mask_gt, aspect='auto')

            axes[3].set_title('Image & Predicted mask')
            axes[3].imshow(img, aspect='auto')
            axes[3].imshow(seg_mask_preds, aspect='auto', alpha=0.5)

            fig.tight_layout()

            # Log plot to tensorboard
            tensorboard = self.logger.experiment[-1]
            tensorboard.add_figure(
                f'{mode}/set{dataloader_idx}_batch{batch_idx}',
                fig,
                self.current_epoch
            )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx, mode='test')

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict
