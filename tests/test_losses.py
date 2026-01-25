"""
损失函数单元测试
"""
import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.losses import DiceLoss, BCEDiceLoss, FocalLoss


class TestDiceLoss:
    """测试Dice损失函数"""

    def test_dice_loss_perfect_prediction(self):
        """测试完美预测的情况"""
        loss_fn = DiceLoss()
        pred = torch.ones(2, 1, 64, 64) * 10.0  # 经过sigmoid后接近1
        target = torch.ones(2, 1, 64, 64)
        loss = loss_fn(pred, target)

        # 完美预测应该接近0
        assert loss.item() < 0.01

    def test_dice_loss_worst_prediction(self):
        """测试最差预测的情况"""
        loss_fn = DiceLoss()
        pred = torch.ones(2, 1, 64, 64) * 10.0  # 经过sigmoid后接近1
        target = torch.zeros(2, 1, 64, 64)
        loss = loss_fn(pred, target)

        # 最差预测应该接近1
        assert loss.item() > 0.9

    def test_dice_loss_shape(self):
        """测试损失函数输出形状"""
        loss_fn = DiceLoss()
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)

        # 损失应该是标量
        assert loss.dim() == 0

    def test_dice_loss_gradient(self):
        """测试梯度计算"""
        loss_fn = DiceLoss()
        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        loss.backward()

        # 检查梯度是否存在
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape


class TestBCEDiceLoss:
    """测试BCE+Dice组合损失函数"""

    def test_bce_dice_loss_forward(self):
        """测试前向传播"""
        loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)

        # 损失应该是标量且为正数
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_bce_dice_loss_weights(self):
        """测试不同权重配置"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        # 只使用BCE
        loss_fn_bce = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)
        loss_bce = loss_fn_bce(pred, target)

        # 只使用Dice
        loss_fn_dice = BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)
        loss_dice = loss_fn_dice(pred, target)

        # 两个损失应该不同
        assert abs(loss_bce.item() - loss_dice.item()) > 0.01

    def test_bce_dice_loss_gradient(self):
        """测试梯度计算"""
        loss_fn = BCEDiceLoss()
        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None


class TestFocalLoss:
    """测试Focal损失函数"""

    def test_focal_loss_forward(self):
        """测试前向传播"""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)

        # 损失应该是标量且为正数
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_focal_loss_parameters(self):
        """测试不同参数配置"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        # 不同gamma值
        loss_fn_1 = FocalLoss(alpha=0.25, gamma=1.0)
        loss_1 = loss_fn_1(pred, target)

        loss_fn_2 = FocalLoss(alpha=0.25, gamma=3.0)
        loss_2 = loss_fn_2(pred, target)

        # 损失值应该不同
        assert abs(loss_1.item() - loss_2.item()) > 0.001

    def test_focal_loss_gradient(self):
        """测试梯度计算"""
        loss_fn = FocalLoss()
        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
