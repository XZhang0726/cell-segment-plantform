"""
U-Net模型单元测试
"""
import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.models.unet import UNet, DoubleConv, Down, Up, OutConv


class TestDoubleConv:
    """测试DoubleConv模块"""

    def test_double_conv_forward(self):
        """测试DoubleConv前向传播"""
        module = DoubleConv(3, 64)
        x = torch.randn(2, 3, 256, 256)
        output = module(x)

        assert output.shape == (2, 64, 256, 256)
        assert output.dtype == torch.float32

    def test_double_conv_with_mid_channels(self):
        """测试带中间通道的DoubleConv"""
        module = DoubleConv(3, 64, mid_channels=32)
        x = torch.randn(2, 3, 256, 256)
        output = module(x)

        assert output.shape == (2, 64, 256, 256)


class TestDown:
    """测试Down模块"""

    def test_down_forward(self):
        """测试Down前向传播"""
        module = Down(64, 128)
        x = torch.randn(2, 64, 256, 256)
        output = module(x)

        # 下采样后尺寸减半
        assert output.shape == (2, 128, 128, 128)


class TestUp:
    """测试Up模块"""

    def test_up_forward_bilinear(self):
        """测试双线性插值上采样"""
        module = Up(128, 64, bilinear=True)
        x1 = torch.randn(2, 128, 64, 64)
        x2 = torch.randn(2, 64, 128, 128)
        output = module(x1, x2)

        assert output.shape == (2, 64, 128, 128)

    def test_up_forward_transpose(self):
        """测试转置卷积上采样"""
        module = Up(128, 64, bilinear=False)
        x1 = torch.randn(2, 128, 64, 64)
        x2 = torch.randn(2, 64, 128, 128)
        output = module(x1, x2)

        assert output.shape == (2, 64, 128, 128)


class TestOutConv:
    """测试OutConv模块"""

    def test_out_conv_forward(self):
        """测试OutConv前向传播"""
        module = OutConv(64, 1)
        x = torch.randn(2, 64, 256, 256)
        output = module(x)

        assert output.shape == (2, 1, 256, 256)


class TestUNet:
    """测试完整的U-Net模型"""

    def test_unet_forward_default(self):
        """测试默认配置的U-Net前向传播"""
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        assert output.shape == (2, 1, 256, 256)
        assert output.dtype == torch.float32

    def test_unet_forward_grayscale(self):
        """测试灰度图输入的U-Net"""
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        x = torch.randn(2, 1, 256, 256)
        output = model(x)

        assert output.shape == (2, 1, 256, 256)

    def test_unet_forward_multiclass(self):
        """测试多类别输出的U-Net"""
        model = UNet(n_channels=3, n_classes=5, bilinear=True)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        assert output.shape == (2, 5, 256, 256)

    def test_unet_forward_transpose_conv(self):
        """测试使用转置卷积的U-Net"""
        model = UNet(n_channels=3, n_classes=1, bilinear=False)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        assert output.shape == (2, 1, 256, 256)

    def test_unet_different_input_sizes(self):
        """测试不同输入尺寸"""
        model = UNet(n_channels=3, n_classes=1, bilinear=True)

        # 测试512x512
        x = torch.randn(1, 3, 512, 512)
        output = model(x)
        assert output.shape == (1, 1, 512, 512)

        # 测试128x128
        x = torch.randn(1, 3, 128, 128)
        output = model(x)
        assert output.shape == (1, 1, 128, 128)

    def test_unet_batch_sizes(self):
        """测试不同批次大小"""
        model = UNet(n_channels=3, n_classes=1, bilinear=True)

        # 批次大小为1
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        assert output.shape == (1, 1, 256, 256)

        # 批次大小为8
        x = torch.randn(8, 3, 256, 256)
        output = model(x)
        assert output.shape == (8, 1, 256, 256)

    def test_unet_gradient_flow(self):
        """测试梯度流动"""
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # 检查输入是否有梯度
        assert x.grad is not None
        assert x.grad.shape == x.shape
