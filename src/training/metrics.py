"""
评估指标模块

提供用于细胞分割的各种评估指标
"""
import torch
import numpy as np
from typing import Dict

from ..core.utils.logger import get_logger

logger = get_logger(__name__)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    计算Dice系数

    Args:
        pred: 预测值 (N, C, H, W)
        target: 目标值 (N, C, H, W)
        smooth: 平滑因子

    Returns:
        Dice系数
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # 展平张量
    pred = pred.view(-1)
    target = target.view(-1)

    # 计算Dice系数
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    计算IoU (Intersection over Union)

    Args:
        pred: 预测值 (N, C, H, W)
        target: 目标值 (N, C, H, W)
        smooth: 平滑因子

    Returns:
        IoU分数
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # 展平张量
    pred = pred.view(-1)
    target = target.view(-1)

    # 计算IoU
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算像素准确率

    Args:
        pred: 预测值 (N, C, H, W)
        target: 目标值 (N, C, H, W)

    Returns:
        像素准确率
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    correct = (pred == target).sum()
    total = target.numel()
    accuracy = correct / total

    return accuracy.item()


def precision_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    计算精确率

    Args:
        pred: 预测值 (N, C, H, W)
        target: 目标值 (N, C, H, W)
        smooth: 平滑因子

    Returns:
        精确率
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # 展平张量
    pred = pred.view(-1)
    target = target.view(-1)

    # 计算精确率
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    precision = (true_positive + smooth) / (predicted_positive + smooth)

    return precision.item()


def recall_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    计算召回率

    Args:
        pred: 预测值 (N, C, H, W)
        target: 目标值 (N, C, H, W)
        smooth: 平滑因子

    Returns:
        召回率
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # 展平张量
    pred = pred.view(-1)
    target = target.view(-1)

    # 计算召回率
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    recall = (true_positive + smooth) / (actual_positive + smooth)

    return recall.item()


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        pred: 预测值 (N, C, H, W)
        target: 目标值 (N, C, H, W)

    Returns:
        包含所有指标的字典
    """
    metrics = {
        'dice': dice_coefficient(pred, target),
        'iou': iou_score(pred, target),
        'accuracy': pixel_accuracy(pred, target),
        'precision': precision_score(pred, target),
        'recall': recall_score(pred, target)
    }

    return metrics


