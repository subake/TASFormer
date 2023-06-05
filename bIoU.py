
import numpy as np
import os
from PIL import Image
import cv2

# annotations
gt_path = "/hdd/wingrune/ADEChallengeData2016/annotations/validation"
# path to masks with all categories
pred_path = "/home/wingrune/cv/TASFormer/ADE20K_multitask_segmentation/results/adapter_150"

num_classes = 150
iou_scores_gt = []
for idx, gt_filename in enumerate(os.listdir(gt_path)):
    gt_mask = np.array(Image.open(f"{gt_path}/{gt_filename}"))
    gt_mask = gt_mask.astype(np.int64)
    gt_idx = gt_filename.split(".")[0]

    for cat in range(1, num_classes+1):

        gt = gt_mask == cat
        if np.sum(gt) > 0:
            pred = np.array(Image.open(f"{pred_path}/{gt_idx}_{cat-1}.png"))
            pred = cv2.resize(pred, dsize=(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred = pred.astype(bool)
            intersection = np.logical_and(gt, pred)
            union = np.logical_or(gt, pred)
            iou_score = np.sum(intersection) / np.sum(union)
            
            iou_scores_gt.append(iou_score)


    print("Done: ", idx)
    print("Mean IoU GT:", np.mean(iou_scores_gt))
print("Mean IoU GT:", np.mean(iou_scores_gt))