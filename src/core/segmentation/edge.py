"""
边缘检测算法模块

提供各种边缘检测方法
"""
import cv2
import numpy as np
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EdgeDetection:
    """边缘检测类"""

    @staticmethod
    def canny(
        image: np.ndarray,
        threshold1: int = 50,
        threshold2: int = 150,
        aperture_size: int = 3,
        L2gradient: bool = False
    ) -> np.ndarray:
        """
        Canny边缘检测

        Args:
            image: 输入图像（灰度图）
            threshold1: 第一个阈值（低阈值）
            threshold2: 第二个阈值（高阈值）
            aperture_size: Sobel算子的孔径大小
            L2gradient: 是否使用L2范数计算梯度幅值

        Returns:
            边缘图像
        """
        if image.ndim != 2:
            raise ValueError("Canny edge detection requires grayscale image")

        edges = cv2.Canny(
            image,
            threshold1,
            threshold2,
            apertureSize=aperture_size,
            L2gradient=L2gradient
        )

        logger.debug(f"Canny edge detection: threshold1={threshold1}, threshold2={threshold2}")
        return edges

    @staticmethod
    def sobel(
        image: np.ndarray,
        dx: int = 1,
        dy: int = 1,
        ksize: int = 3
    ) -> np.ndarray:
        """
        Sobel边缘检测

        Args:
            image: 输入图像（灰度图）
            dx: x方向的导数阶数
            dy: y方向的导数阶数
            ksize: Sobel核的大小

        Returns:
            边缘图像
        """
        if image.ndim != 2:
            raise ValueError("Sobel edge detection requires grayscale image")

        # 计算x和y方向的梯度
        if dx > 0:
            grad_x = cv2.Sobel(image, cv2.CV_64F, dx, 0, ksize=ksize)
            grad_x = np.abs(grad_x)
        else:
            grad_x = 0

        if dy > 0:
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, dy, ksize=ksize)
            grad_y = np.abs(grad_y)
        else:
            grad_y = 0

        # 合并梯度
        if dx > 0 and dy > 0:
            edges = np.sqrt(grad_x**2 + grad_y**2)
        elif dx > 0:
            edges = grad_x
        else:
            edges = grad_y

        # 归一化到0-255
        edges = np.uint8(np.clip(edges, 0, 255))

        logger.debug(f"Sobel edge detection: dx={dx}, dy={dy}, ksize={ksize}")
        return edges

    @staticmethod
    def laplacian(
        image: np.ndarray,
        ksize: int = 3
    ) -> np.ndarray:
        """
        Laplacian边缘检测

        Args:
            image: 输入图像（灰度图）
            ksize: 核大小

        Returns:
            边缘图像
        """
        if image.ndim != 2:
            raise ValueError("Laplacian edge detection requires grayscale image")

        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
        laplacian = np.abs(laplacian)
        laplacian = np.uint8(np.clip(laplacian, 0, 255))

        logger.debug(f"Laplacian edge detection: ksize={ksize}")
        return laplacian

    @staticmethod
    def scharr(
        image: np.ndarray,
        dx: int = 1,
        dy: int = 0
    ) -> np.ndarray:
        """
        Scharr边缘检测（更精确的Sobel算子）

        Args:
            image: 输入图像（灰度图）
            dx: x方向的导数阶数
            dy: y方向的导数阶数

        Returns:
            边缘图像
        """
        if image.ndim != 2:
            raise ValueError("Scharr edge detection requires grayscale image")

        scharr = cv2.Scharr(image, cv2.CV_64F, dx, dy)
        scharr = np.abs(scharr)
        scharr = np.uint8(np.clip(scharr, 0, 255))

        logger.debug(f"Scharr edge detection: dx={dx}, dy={dy}")
        return scharr


# 便捷函数
def canny_edge(image: np.ndarray, threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
    """Canny边缘检测的便捷函数"""
    return EdgeDetection.canny(image, threshold1, threshold2)


def sobel_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Sobel边缘检测的便捷函数"""
    return EdgeDetection.sobel(image, dx=1, dy=1, ksize=ksize)


def laplacian_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Laplacian边缘检测的便捷函数"""
    return EdgeDetection.laplacian(image, ksize=ksize)
