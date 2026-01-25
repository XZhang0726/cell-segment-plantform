"""
评估指标单元测试
"""
import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.metrics import (
    dice_coefficient,
    iou_score,
    pixel_accuracy,
    precision_score,
    recall_score,
    calculate_metrics
)


class TestDiceCoefficient:
    """测试Dice系数"""

    def test_dice_perfect_prediction(self):
        """测试完美预测"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        dice = dice_coefficient(pred, target)

        # 完美预测应该接近1
        assert dice > 0.99

    def test_dice_worst_prediction(self):
        """测试最差预测"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.zeros(2, 1, 64, 64)
        dice = dice_coefficient(pred, target)

        # 最差预测应该接近0
        assert dice < 0.01

    def test_dice_return_type(self):
        """测试返回类型"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        dice = dice_coefficient(pred, target)

        # 应该返回float
        assert isinstance(dice, float)
        assert 0 <= dice <= 1


class TestIoUScore:
    """测试IoU分数"""

    def test_iou_perfect_prediction(self):
        """测试完美预测"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        iou = iou_score(pred, target)

        # 完美预测应该接近1
        assert iou > 0.99

    def test_iou_worst_prediction(self):
        """测试最差预测"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.zeros(2, 1, 64, 64)
        iou = iou_score(pred, target)

        # 最差预测应该接近0
        assert iou < 0.01

    def test_iou_return_type(self):
        """测试返回类型"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        iou = iou_score(pred, target)

        assert isinstance(iou, float)
        assert 0 <= iou <= 1


class TestPixelAccuracy:
    """测试像素准确率"""

    def test_accuracy_perfect_prediction(self):
        """测试完美预测"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        acc = pixel_accuracy(pred, target)

        # 完美预测应该接近1
        assert acc > 0.99

    def test_accuracy_return_type(self):
        """测试返回类型"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        acc = pixel_accuracy(pred, target)

        assert isinstance(acc, float)
        assert 0 <= acc <= 1


class TestPrecisionScore:
    """测试精确率"""

    def test_precision_perfect_prediction(self):
        """测试完美预测"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        precision = precision_score(pred, target)

        assert precision > 0.99

    def test_precision_return_type(self):
        """测试返回类型"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        precision = precision_score(pred, target)

        assert isinstance(precision, float)
        assert 0 <= precision <= 1


class TestRecallScore:
    """测试召回率"""

    def test_recall_perfect_prediction(self):
        """测试完美预测"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        recall = recall_score(pred, target)

        assert recall > 0.99

    def test_recall_return_type(self):
        """测试返回类型"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        recall = recall_score(pred, target)

        assert isinstance(recall, float)
        assert 0 <= recall <= 1


class TestCalculateMetrics:
    """测试综合指标计算"""

    def test_calculate_metrics_return_type(self):
        """测试返回类型"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        metrics = calculate_metrics(pred, target)

        # 应该返回字典
        assert isinstance(metrics, dict)

        # 检查所有指标是否存在
        expected_keys = ['dice', 'iou', 'accuracy', 'precision', 'recall']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
            assert 0 <= metrics[key] <= 1

    def test_calculate_metrics_perfect_prediction(self):
        """测试完美预测的所有指标"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        metrics = calculate_metrics(pred, target)

        # 所有指标都应该接近1
        for key, value in metrics.items():
            assert value > 0.99, f"{key} should be close to 1 for perfect prediction"
