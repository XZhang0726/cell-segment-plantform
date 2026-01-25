"""
数据增强模块

提供用于细胞分割的数据增强变换
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional

from ..core.utils.logger import get_logger

logger = get_logger(__name__)


def get_training_augmentation(image_size: tuple = (256, 256)):
    """
    获取训练时的数据增强

    Args:
        image_size: 目标图像尺寸 (height, width)

    Returns:
        albumentations变换组合
    """
    train_transform = A.Compose([
        # 几何变换
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        ),

        # 弹性变形（对细胞分割很有用）
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.3
        ),

        # 网格扭曲
        A.GridDistortion(p=0.3),
    ])

    logger.info(f"Created training augmentation pipeline for size {image_size}")
    return train_transform


def get_validation_augmentation(image_size: tuple = (256, 256)):
    """
    获取验证时的数据增强（仅调整大小）

    Args:
        image_size: 目标图像尺寸 (height, width)

    Returns:
        albumentations变换组合
    """
    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
    ])

    logger.info(f"Created validation augmentation pipeline for size {image_size}")
    return val_transform


def get_test_augmentation(image_size: tuple = (256, 256)):
    """
    获取测试时的数据增强（仅调整大小）

    Args:
        image_size: 目标图像尺寸 (height, width)

    Returns:
        albumentations变换组合
    """
    test_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
    ])

    logger.info(f"Created test augmentation pipeline for size {image_size}")
    return test_transform

