# mIoU computation is taken from detectron2
# https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/sem_seg_evaluation.html#SemSegEvaluator.evaluate

import numpy as np
import os
from PIL import Image
import cv2

# annotations
gt_path = "/hdd/wingrune/ADEChallengeData2016/annotations/validation"
# path to masks with all categories
pred_path = "/home/wingrune/cv/TASFormer/ADE20K_multitask_segmentation/results/adapter_150_640"

num_classes = 150

conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
for idx, gt_filename in enumerate(os.listdir(gt_path)):
    #gt_filename = 'ADE_val_00000001.png'
    gt = np.array(Image.open(f"{gt_path}/{gt_filename}"))
    gt = gt -1
    gt = gt.astype(np.int64)
    gt_idx = gt_filename.split(".")[0]
    #print(np.unique(gt))
    pred = np.load(f"{pred_path}/{gt_idx}.npy")
    #print(np.unique(pred))
    pred = cv2.resize(pred, dsize=(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred = pred.astype(np.int64)
    gt[gt == 255] = num_classes
    conf_matrix += np.bincount(
        (num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
        minlength=conf_matrix.size,
    ).reshape(conf_matrix.shape)


acc = np.full(num_classes, np.nan, dtype=np.float)
iou = np.full(num_classes, np.nan, dtype=np.float)
tp = conf_matrix.diagonal()[:-1].astype(np.float)
pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float)
class_weights = pos_gt / np.sum(pos_gt)
pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float)
acc_valid = pos_gt > 0
acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
union = pos_gt + pos_pred - tp
iou_valid = np.logical_and(acc_valid, union > 0)
iou[iou_valid] = tp[iou_valid] / union[iou_valid]
macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
pacc = np.sum(tp) / np.sum(pos_gt)

res = {}
res["mIoU"] = 100 * miou
res["fwIoU"] = 100 * fiou

res["mACC"] = 100 * macc
res["pACC"] = 100 * pacc


print("mIoU: ",  res["mIoU"], "fwIoU: ", res["fwIoU"], "mACC: ", res["mACC"], "pACC: ", res["pACC"])