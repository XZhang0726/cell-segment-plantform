"""
阈值分割算法模块

提供各种阈值分割方法
"""
import cv2
import numpy as np
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ThresholdSegmentation:
    """阈值分割类"""

    @staticmethod
    def otsu_threshold(
        image: np.ndarray,
        return_threshold: bool = False
    ) -> np.ndarray:
        """
        Otsu自动阈值分割

        Args:
            image: 输入图像（灰度图）
            return_threshold: 是否返回阈值

        Returns:
            二值化图像，如果return_threshold=True则返回(二值图, 阈值)
        """
        if image.ndim != 2:
            raise ValueError("Otsu threshold requires grayscale image")

        # 使用Otsu方法自动计算阈值
        threshold_value, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        logger.debug(f"Otsu threshold value: {threshold_value:.2f}")

        if return_threshold:
            return binary, threshold_value
        return binary

    @staticmethod
    def fixed_threshold(
        image: np.ndarray,
        threshold: int = 127,
        max_value: int = 255,
        threshold_type: str = 'binary'
    ) -> np.ndarray:
        """
        固定阈值分割

        Args:
            image: 输入图像（灰度图）
            threshold: 阈值
            max_value: 最大值
            threshold_type: 阈值类型 ('binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv')

        Returns:
            二值化图像
        """
        if image.ndim != 2:
            raise ValueError("Fixed threshold requires grayscale image")

        threshold_types = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
            'trunc': cv2.THRESH_TRUNC,
            'tozero': cv2.THRESH_TOZERO,
            'tozero_inv': cv2.THRESH_TOZERO_INV
        }

        if threshold_type not in threshold_types:
            raise ValueError(f"Unknown threshold type: {threshold_type}")

        _, binary = cv2.threshold(
            image, threshold, max_value, threshold_types[threshold_type]
        )

        logger.debug(f"Fixed threshold: {threshold}, type: {threshold_type}")
        return binary

    @staticmethod
    def adaptive_threshold(
        image: np.ndarray,
        max_value: int = 255,
        method: str = 'gaussian',
        threshold_type: str = 'binary',
        block_size: int = 11,
        C: int = 2
    ) -> np.ndarray:
        """
        自适应阈值分割

        Args:
            image: 输入图像（灰度图）
            max_value: 最大值
            method: 自适应方法 ('mean', 'gaussian')
            threshold_type: 阈值类型 ('binary', 'binary_inv')
            block_size: 邻域大小（必须是奇数）
            C: 常数，从计算的平均值或加权平均值中减去

        Returns:
            二值化图像
        """
        if image.ndim != 2:
            raise ValueError("Adaptive threshold requires grayscale image")

        if block_size % 2 == 0:
            block_size += 1
            logger.warning(f"Block size must be odd, adjusted to {block_size}")

        adaptive_methods = {
            'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
            'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        }

        threshold_types = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV
        }

        if method not in adaptive_methods:
            raise ValueError(f"Unknown adaptive method: {method}")

        if threshold_type not in threshold_types:
            raise ValueError(f"Unknown threshold type: {threshold_type}")

        binary = cv2.adaptiveThreshold(
            image,
            max_value,
            adaptive_methods[method],
            threshold_types[threshold_type],
            block_size,
            C
        )

        logger.debug(f"Adaptive threshold: method={method}, block_size={block_size}")
        return binary

    @staticmethod
    def multi_otsu(
        image: np.ndarray,
        n_classes: int = 3
    ) -> np.ndarray:
        """
        多阈值Otsu分割

        Args:
            image: 输入图像（灰度图）
            n_classes: 分类数量

        Returns:
            分割后的图像
        """
        if image.ndim != 2:
            raise ValueError("Multi-Otsu requires grayscale image")

        from skimage.filters import threshold_multiotsu

        # 计算多个阈值
        thresholds = threshold_multiotsu(image, classes=n_classes)

        # 根据阈值分割图像
        segmented = np.digitize(image, bins=thresholds)

        logger.debug(f"Multi-Otsu thresholds: {thresholds}")
        return segmented.astype(np.uint8)


# 便捷函数
def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Otsu阈值分割的便捷函数"""
    return ThresholdSegmentation.otsu_threshold(image)


def adaptive_threshold(image: np.ndarray, block_size: int = 11) -> np.ndarray:
    """自适应阈值分割的便捷函数"""
    return ThresholdSegmentation.adaptive_threshold(image, block_size=block_size)


def fixed_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """固定阈值分割的便捷函数"""
    return ThresholdSegmentation.fixed_threshold(image, threshold=threshold)
