import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def calculate_metrics(pred, target):
    """计算评估指标"""
    pred_bin = (pred > 0.5).float()

    # 计算基础指标
    tp = torch.sum(pred_bin * target)
    fp = torch.sum(pred_bin * (1 - target))
    tn = torch.sum((1 - pred_bin) * (1 - target))
    fn = torch.sum((1 - pred_bin) * target)

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    intersection = torch.sum(pred_bin * target)
    union = torch.sum(pred_bin) + torch.sum(target) - intersection
    iou = intersection / (union + 1e-8)

    precision = tp / (tp + fp + 1e-8)

    # 计算95% HD和ASD
    try:
        pred_np = pred_bin.squeeze().cpu().numpy().astype(np.uint8)
        target_np = target.squeeze().cpu().numpy().astype(np.uint8)

        if np.sum(pred_np) == 0 and np.sum(target_np) == 0:
            hd95_value = 0.0
            asd_value = 0.0
        elif np.sum(pred_np) > 0 and np.sum(target_np) > 0:
            # 计算距离变换
            pred_dist = distance_transform_edt(1 - pred_np)
            target_dist = distance_transform_edt(1 - target_np)

            # 获取边界距离
            pred_to_target = pred_dist[target_np > 0]
            target_to_pred = target_dist[pred_np > 0]

            # 合并距离并计算95%分位数
            all_distances = np.concatenate([pred_to_target, target_to_pred])
            hd95_value = np.percentile(all_distances, 95) if len(all_distances) > 0 else 0.0
            asd_value = (np.mean(pred_to_target) + np.mean(target_to_pred)) / 2 if len(all_distances) > 0 else 0.0
        else:
            # 一方为空时使用图像对角线作为最大距离
            max_dist = np.sqrt(pred_np.shape[0] ** 2 + pred_np.shape[1] ** 2)
            hd95_value = max_dist
            asd_value = max_dist
    except Exception as e:
        hd95_value = 0.0
        asd_value = 0.0

    return dice.item(), sensitivity.item(), specificity.item(), iou.item(), precision.item(), hd95_value, asd_value