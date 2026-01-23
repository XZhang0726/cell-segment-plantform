"""
图像预处理模块

提供常用的图像预处理功能
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from skimage import exposure, filters

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """图像预处理类"""

    @staticmethod
    def normalize(
        image: np.ndarray,
        method: str = 'minmax',
        clip_percentile: Tuple[float, float] = (1, 99)
    ) -> np.ndarray:
        """
        图像归一化

        Args:
            image: 输入图像
            method: 归一化方法 ('minmax', 'zscore', 'percentile')
            clip_percentile: 百分位裁剪范围 (仅用于percentile方法)

        Returns:
            归一化后的图像
        """
        image = image.astype(np.float32)

        if method == 'minmax':
            # Min-Max归一化到[0, 1]
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = image

        elif method == 'zscore':
            # Z-score标准化
            mean = image.mean()
            std = image.std()
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image - mean

        elif method == 'percentile':
            # 百分位裁剪后归一化
            p_low, p_high = np.percentile(image, clip_percentile)
            image_clipped = np.clip(image, p_low, p_high)
            normalized = (image_clipped - p_low) / (p_high - p_low)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        logger.debug(f"Normalized image using {method} method")
        return normalized

    @staticmethod
    def resize(
        image: np.ndarray,
        size: Tuple[int, int],
        interpolation: str = 'bilinear'
    ) -> np.ndarray:
        """
        调整图像大小

        Args:
            image: 输入图像
            size: 目标大小 (width, height)
            interpolation: 插值方法 ('nearest', 'bilinear', 'bicubic', 'lanczos')

        Returns:
            调整大小后的图像
        """
        interp_methods = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }

        if interpolation not in interp_methods:
            raise ValueError(f"Unknown interpolation method: {interpolation}")

        resized = cv2.resize(image, size, interpolation=interp_methods[interpolation])
        logger.debug(f"Resized image to {size}")
        return resized

    @staticmethod
    def denoise(
        image: np.ndarray,
        method: str = 'gaussian',
        **kwargs
    ) -> np.ndarray:
        """
        图像去噪

        Args:
            image: 输入图像
            method: 去噪方法 ('gaussian', 'median', 'bilateral', 'nlm')
            **kwargs: 方法特定的参数

        Returns:
            去噪后的图像
        """
        if method == 'gaussian':
            # 高斯滤波
            ksize = kwargs.get('ksize', 5)
            sigma = kwargs.get('sigma', 0)
            denoised = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        elif method == 'median':
            # 中值滤波
            ksize = kwargs.get('ksize', 5)
            denoised = cv2.medianBlur(image, ksize)

        elif method == 'bilateral':
            # 双边滤波
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        elif method == 'nlm':
            # 非局部均值去噪
            h = kwargs.get('h', 10)
            template_window_size = kwargs.get('template_window_size', 7)
            search_window_size = kwargs.get('search_window_size', 21)

            if image.ndim == 2:
                denoised = cv2.fastNlMeansDenoising(
                    image, None, h, template_window_size, search_window_size
                )
            else:
                denoised = cv2.fastNlMeansDenoisingColored(
                    image, None, h, h, template_window_size, search_window_size
                )

        else:
            raise ValueError(f"Unknown denoising method: {method}")

        logger.debug(f"Denoised image using {method} method")
        return denoised

    @staticmethod
    def enhance_contrast(
        image: np.ndarray,
        method: str = 'clahe',
        **kwargs
    ) -> np.ndarray:
        """
        对比度增强

        Args:
            image: 输入图像
            method: 增强方法 ('clahe', 'histogram_eq', 'adaptive_eq')
            **kwargs: 方法特定的参数

        Returns:
            增强后的图像
        """
        if method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clip_limit = kwargs.get('clip_limit', 2.0)
            tile_grid_size = kwargs.get('tile_grid_size', (8, 8))

            if image.ndim == 2:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                enhanced = clahe.apply(image)
            else:
                # 对彩色图像，在LAB空间的L通道应用CLAHE
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        elif method == 'histogram_eq':
            # 直方图均衡化
            if image.ndim == 2:
                enhanced = cv2.equalizeHist(image)
            else:
                # 对彩色图像，在YUV空间的Y通道应用
                yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

        elif method == 'adaptive_eq':
            # 自适应直方图均衡化
            clip_limit = kwargs.get('clip_limit', 0.03)
            enhanced = exposure.equalize_adapthist(image, clip_limit=clip_limit)
            # 转换回原始数据类型范围
            if image.dtype == np.uint8:
                enhanced = (enhanced * 255).astype(np.uint8)

        else:
            raise ValueError(f"Unknown contrast enhancement method: {method}")

        logger.debug(f"Enhanced contrast using {method} method")
        return enhanced


# 便捷函数
def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """归一化图像的便捷函数"""
    return ImagePreprocessor.normalize(image, method=method)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """调整图像大小的便捷函数"""
    return ImagePreprocessor.resize(image, size)


def denoise_image(image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
    """图像去噪的便捷函数"""
    return ImagePreprocessor.denoise(image, method=method)


def enhance_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """对比度增强的便捷函数"""
    return ImagePreprocessor.enhance_contrast(image, method=method)
