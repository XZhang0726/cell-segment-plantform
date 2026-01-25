"""
细胞分割数据集加载器

提供PyTorch Dataset类用于加载图像和掩码
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Tuple

from .image_io import ImageIO
from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class CellSegmentationDataset(Dataset):
    """
    细胞分割数据集

    Args:
        image_dir: 图像目录路径
        mask_dir: 掩码目录路径
        transform: 数据增强变换
        image_size: 目标图像尺寸 (height, width)
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        image_size: Optional[Tuple[int, int]] = None
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size

        # 获取所有图像文件
        self.image_files = self._get_image_files()

        logger.info(f"Loaded dataset: {len(self.image_files)} images from {image_dir}")

    def _get_image_files(self) -> List[Path]:
        """获取所有图像文件"""
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        image_files = []

        for ext in image_extensions:
            image_files.extend(self.image_dir.glob(f'*{ext}'))
            image_files.extend(self.image_dir.glob(f'*{ext.upper()}'))

        return sorted(image_files)

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (image, mask) 元组
        """
        # 加载图像
        image_path = self.image_files[idx]
        image = ImageIO.load_image(image_path)

        # 加载对应的掩码
        mask_path = self.mask_dir / image_path.name
        if not mask_path.exists():
            # 尝试其他可能的掩码文件名
            mask_path = self.mask_dir / (image_path.stem + '_mask' + image_path.suffix)

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image: {image_path}")

        mask = ImageIO.load_image(mask_path, grayscale=True)

        # 调整图像大小
        if self.image_size is not None:
            import cv2
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))

        # 应用数据增强
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # 转换为张量
        image = self._to_tensor(image)
        mask = self._to_tensor(mask)

        return image, mask

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        将numpy数组转换为PyTorch张量

        Args:
            image: numpy数组 (H, W) 或 (H, W, C)

        Returns:
            PyTorch张量 (C, H, W) 或 (1, H, W)
        """
        # 归一化到[0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # 转换维度顺序
        if image.ndim == 2:
            # 灰度图 (H, W) -> (1, H, W)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:
            # 彩色图 (H, W, C) -> (C, H, W)
            image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image.copy()).float()

