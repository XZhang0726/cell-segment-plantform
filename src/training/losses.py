"""
损失函数模块

提供用于细胞分割的各种损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class DiceLoss(nn.Module):
    """
    Dice损失函数

    适用于分割任务，特别是处理类别不平衡问题
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Dice损失

        Args:
            pred: 预测值 (N, C, H, W)
            target: 目标值 (N, C, H, W)

        Returns:
            Dice损失值
        """
        pred = torch.sigmoid(pred)

        # 展平张量
        pred = pred.view(-1)
        target = target.view(-1)

        # 计算Dice系数
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice组合损失函数

    结合二元交叉熵和Dice损失的优点
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失

        Args:
            pred: 预测值 (N, C, H, W)
            target: 目标值 (N, C, H, W)

        Returns:
            组合损失值
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal损失函数

    用于处理类别不平衡问题，关注难分类样本
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Focal损失

        Args:
            pred: 预测值 (N, C, H, W)
            target: 目标值 (N, C, H, W)

        Returns:
            Focal损失值
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()

