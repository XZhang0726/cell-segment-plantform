"""
形态学操作模块

提供各种形态学操作方法
"""
import cv2
import numpy as np
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MorphologicalOps:
    """形态学操作类"""

    @staticmethod
    def get_kernel(shape: str = 'rect', size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """
        获取形态学核

        Args:
            shape: 核形状 ('rect', 'ellipse', 'cross')
            size: 核大小

        Returns:
            形态学核
        """
        shapes = {
            'rect': cv2.MORPH_RECT,
            'ellipse': cv2.MORPH_ELLIPSE,
            'cross': cv2.MORPH_CROSS
        }

        if shape not in shapes:
            raise ValueError(f"Unknown kernel shape: {shape}")

        kernel = cv2.getStructuringElement(shapes[shape], size)
        return kernel

    @staticmethod
    def erode(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect',
        iterations: int = 1
    ) -> np.ndarray:
        """
        腐蚀操作

        Args:
            image: 输入图像
            kernel_size: 核大小
            kernel_shape: 核形状
            iterations: 迭代次数

        Returns:
            腐蚀后的图像
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        eroded = cv2.erode(image, kernel, iterations=iterations)

        logger.debug(f"Erosion: kernel_size={kernel_size}, iterations={iterations}")
        return eroded

    @staticmethod
    def dilate(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect',
        iterations: int = 1
    ) -> np.ndarray:
        """
        膨胀操作

        Args:
            image: 输入图像
            kernel_size: 核大小
            kernel_shape: 核形状
            iterations: 迭代次数

        Returns:
            膨胀后的图像
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        dilated = cv2.dilate(image, kernel, iterations=iterations)

        logger.debug(f"Dilation: kernel_size={kernel_size}, iterations={iterations}")
        return dilated

    @staticmethod
    def opening(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect'
    ) -> np.ndarray:
        """
        开运算（先腐蚀后膨胀，去除小物体）

        Args:
            image: 输入图像
            kernel_size: 核大小
            kernel_shape: 核形状

        Returns:
            开运算后的图像
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        logger.debug(f"Opening: kernel_size={kernel_size}")
        return opened

    @staticmethod
    def closing(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect'
    ) -> np.ndarray:
        """
        闭运算（先膨胀后腐蚀，填充小孔）

        Args:
            image: 输入图像
            kernel_size: 核大小
            kernel_shape: 核形状

        Returns:
            闭运算后的图像
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        logger.debug(f"Closing: kernel_size={kernel_size}")
        return closed

    @staticmethod
    def gradient(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect'
    ) -> np.ndarray:
        """
        形态学梯度（膨胀-腐蚀，提取边界）

        Args:
            image: 输入图像
            kernel_size: 核大小
            kernel_shape: 核形状

        Returns:
            梯度图像
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

        logger.debug(f"Morphological gradient: kernel_size={kernel_size}")
        return gradient

    @staticmethod
    def tophat(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect'
    ) -> np.ndarray:
        """
        顶帽变换（原图-开运算，提取亮细节）

        Args:
            image: 输入图像
            kernel_size: 核大小
            kernel_shape: 核形状

        Returns:
            顶帽变换后的图像
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

        logger.debug(f"Top-hat transform: kernel_size={kernel_size}")
        return tophat

    @staticmethod
    def blackhat(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect'
    ) -> np.ndarray:
        """
        黑帽变换（闭运算-原图，提取暗细节）

        Args:
            image: 输入图像
            kernel_size: 核大小
            kernel_shape: 核形状

        Returns:
            黑帽变换后的图像
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

        logger.debug(f"Black-hat transform: kernel_size={kernel_size}")
        return blackhat


# 便捷函数
def erode(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """腐蚀操作的便捷函数"""
    return MorphologicalOps.erode(image, kernel_size)


def dilate(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """膨胀操作的便捷函数"""
    return MorphologicalOps.dilate(image, kernel_size)


def opening(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """开运算的便捷函数"""
    return MorphologicalOps.opening(image, kernel_size)


def closing(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """闭运算的便捷函数"""
    return MorphologicalOps.closing(image, kernel_size)
