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

        if self.hparams.backbone == 'two_headed_resnet':
            self.mask_net = networks.two_headed_fcn_resnet50.SemSegResNet50(
                num_classes=num_output_channels,
                dropout=self.hparams.dropout,
            )
        elif self.hparams.backbone == 'two_headed_resnet_aux':
            self.mask_net = networks.two_headed_fcn_resnet50_aux_heads.SemSegResNet50(
                num_classes=num_output_channels,
                dropout=self.hparams.dropout,
            )
        elif self.hparams.backbone == 'two_headed_segformer':
            self.mask_net = networks.two_headed_segformer.SegFormer(
                num_classes=num_output_channels,
            )
        elif self.hparams.backbone == 'segformer_with_task_embedding':
            self.mask_net = networks.segformer_with_task_embedding.SegFormer(
                num_classes=num_output_channels,
                num_tasks=self.hparams.num_tasks,
            )
        elif self.hparams.backbone == 'segformer_with_vsa_task_embedding':
            self.mask_net = networks.segformer_with_vsa_task_embedding.SegFormer(
                num_classes=num_output_channels,
            )
        elif self.hparams.backbone == 'segformer_with_adapter':
            self.mask_net = networks.segformer_with_adapter.SegFormer(
                num_classes=num_output_channels,
            )
        elif self.hparams.backbone == 'segformer':
            self.mask_net = networks.segformer.SegFormer(
                num_classes=num_output_channels,
            )
        elif self.hparams.backbone == 'resnet':
            self.mask_net = networks.fcn_resnet50.SemSegResNet50(
                num_classes=num_output_channels,
                dropout=self.hparams.dropout,
            )
        elif self.hparams.backbone == 'espnet':
            self.mask_net = networks.espnet.ESPNet(
                classes=num_output_channels,
                encoderFile='/home/ML/data/GTA5/espnet_p_2_q_8.pth',
            )
        elif  self.hparams.backbone == 'mirrornet': 
            self.mask_net = networks.mirrornet.MirrorNet(
                backbone_path='/home/data/MSD/resnext_101_32x4d.pth',
            )
        elif self.hparams.backbone == 'liteseg_m':
            self.mask_net = networks.liteseg_mobilenet.LiteSegMobileNet(
                n_classes=num_output_channels,
                pretrained=False,
            )  
        elif self.hparams.backbone == 'fastscnn':
            self.mask_net = networks.fast_scnn.FSCNN(
                pretrained=False,
                root=None,
                num_classes=num_output_channels,
            ) 
        elif self.hparams.backbone == 'trans2seg':
            self.mask_net = networks.trans2seg.Trans2Seg(
                pretrained=False,
                root=None,
                num_classes=num_output_channels,
            )
        elif self.hparams.backbone == 'translab':
            self.mask_net = networks.translab.TransLab(
                pretrained=False,
                root=None,
                num_classes=num_output_channels,
            )
        elif self.hparams.backbone == 'deeplabv3_plus':
            self.mask_net = networks.deeplabv3_plus.Deeplabv3_plus(
                num_classes=num_output_channels,
                backend=self.hparams.backend,
                dropout=self.hparams.dropout,
            ) 
        else:
            raise BaseException('backbone must be [resnet, espnet, mirrornet, liteseg_m, fastscnn, trans2seg, translab, pspnet, deeplabv3_plus]')

        # Losses
        if num_output_channels <= 2:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        if self.hparams.backbone == 'trans2seg':
            self.loss_fn = get_segmentation_loss()
        
        if self.hparams.backbone == 'translab':
            self.loss_fn = get_segmentation_loss(model='translab')

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
        if self.hparams.backbone == 'segformer_with_task_embedding' \
            or self.hparams.backbone == 'segformer_with_vsa_task_embedding' \
            or self.hparams.backbone == 'segformer_with_adapter':
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
                if self.hparams.backbone == 'segformer_with_task_embedding' \
                    or self.hparams.backbone == 'segformer_with_vsa_task_embedding' \
                    or self.hparams.backbone == 'segformer_with_adapter':
                    
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
            if self.hparams.backbone == 'segformer_with_task_embedding' \
                or self.hparams.backbone == 'segformer_with_vsa_task_embedding' \
                or self.hparams.backbone == 'segformer_with_adapter':
                outp = self.mask_net(cur_batch['image'], cur_task - 1)
                mask_logits = outp['out'][0]
            else:
                outp = self.mask_net(cur_batch['image'])
                mask_logits = outp['out'][cur_task]

            if (self.hparams.backbone == 'trans2seg') or (self.hparams.backbone == 'translab'):
                loss = self.loss_fn(mask_logits[cur_task], cur_batch['mask'].long())['loss']
            elif cat_num <= 2:
                cur_loss = self.loss_fn(mask_logits[:, 0], cur_batch['mask'].to(torch.float32))
            else:
                cur_loss = self.loss_fn(mask_logits[cur_task], cur_batch['mask'].long())
            loss += cur_loss

            if self.hparams.backbone == 'two_headed_resnet_aux':
                aux_mask_logits = outp['aux'][cur_task]
                if cat_num <= 2:
                    cur_aux_loss = self.loss_fn(aux_mask_logits[:, 0], cur_batch['mask'].to(torch.float32))
                else:
                    cur_loss = self.loss_fn(aux_mask_logits, cur_batch['mask'].long())
                loss += 0.7 * cur_aux_loss

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
            if self.hparams.backbone == 'segformer_with_task_embedding' \
                or self.hparams.backbone == 'segformer_with_vsa_task_embedding' \
                or self.hparams.backbone == 'segformer_with_adapter':
                outp = self.mask_net(batch['image'], cur_task - 1)
                mask_logits = outp['out'][0]
            else:
                outp = self.mask_net(batch['image'])
                mask_logits = outp['out'][cur_task]

            mask = mask_logits
        
        if (self.hparams.backbone == 'trans2seg') or (self.hparams.backbone == 'translab'):
            mask_logits_list = self.mask_net(batch['image'])
            loss = self.loss_fn(mask_logits_list, batch['mask'].long())['loss']

            # Compute IoU metric
            mask_iou = mask.detach().cpu().to(torch.float32)
            batch_iou = batch['mask'].detach().cpu().to(torch.int64)
        elif cat_num <= 2:
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
