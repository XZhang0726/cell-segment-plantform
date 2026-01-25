"""
数据集加载器单元测试
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path
import tempfile
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import CellSegmentationDataset


class TestCellSegmentationDataset:
    """测试细胞分割数据集类"""

    def create_test_images(self, tmpdir, num_images=5):
        """创建测试图像"""
        image_dir = Path(tmpdir) / "images"
        mask_dir = Path(tmpdir) / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_images):
            # 创建RGB图像
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(image).save(image_dir / f"image_{i}.png")

            # 创建灰度掩码
            mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
            Image.fromarray(mask, mode='L').save(mask_dir / f"image_{i}.png")

        return image_dir, mask_dir

    def test_dataset_initialization(self):
        """测试数据集初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir = self.create_test_images(tmpdir, num_images=5)

            dataset = CellSegmentationDataset(
                image_dir=str(image_dir),
                mask_dir=str(mask_dir)
            )

            # 检查数据集大小
            assert len(dataset) == 5

    def test_dataset_len(self):
        """测试数据集长度"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir = self.create_test_images(tmpdir, num_images=10)

            dataset = CellSegmentationDataset(
                image_dir=str(image_dir),
                mask_dir=str(mask_dir)
            )

            assert len(dataset) == 10

    def test_dataset_getitem(self):
        """测试获取单个样本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir = self.create_test_images(tmpdir, num_images=5)

            dataset = CellSegmentationDataset(
                image_dir=str(image_dir),
                mask_dir=str(mask_dir)
            )

            # 获取第一个样本
            image, mask = dataset[0]

            # 检查返回类型
            assert isinstance(image, torch.Tensor)
            assert isinstance(mask, torch.Tensor)

            # 检查张量形状 (C, H, W)
            assert image.dim() == 3
            assert mask.dim() == 3
            assert image.shape[0] == 3  # RGB图像
            assert mask.shape[0] == 1   # 单通道掩码

    def test_dataset_with_image_size(self):
        """测试指定图像尺寸"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir = self.create_test_images(tmpdir, num_images=3)

            dataset = CellSegmentationDataset(
                image_dir=str(image_dir),
                mask_dir=str(mask_dir),
                image_size=(128, 128)
            )

            image, mask = dataset[0]

            # 检查调整后的尺寸
            assert image.shape == (3, 128, 128)
            assert mask.shape == (1, 128, 128)
