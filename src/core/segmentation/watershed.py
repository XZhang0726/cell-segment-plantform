"""
分水岭分割算法模块

提供分水岭分割方法，特别适用于分离粘连的细胞
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage as ndi

from ..utils.logger import get_logger

logger = get_logger(__name__)


class WatershedSegmentation:
    """分水岭分割类"""

    @staticmethod
    def watershed_basic(
        image: np.ndarray,
        markers: np.ndarray
    ) -> np.ndarray:
        """
        基本分水岭分割

        Args:
            image: 输入图像（灰度图或彩色图）
            markers: 标记图像，不同区域用不同的正整数标记

        Returns:
            分割后的标记图像
        """
        # 如果是灰度图，转换为彩色图（OpenCV watershed需要3通道）
        if image.ndim == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()

        # 确保markers是int32类型
        markers = markers.astype(np.int32)

        # 执行分水岭算法
        markers = cv2.watershed(image_color, markers)

        logger.debug(f"Watershed segmentation: {len(np.unique(markers))} regions")
        return markers

    @staticmethod
    def watershed_distance_transform(
        binary_image: np.ndarray,
        min_distance: int = 10,
        return_markers: bool = False
    ) -> np.ndarray:
        """
        基于距离变换的分水岭分割（适用于分离粘连细胞）

        Args:
            binary_image: 二值图像（前景为白色）
            min_distance: 局部最大值之间的最小距离
            return_markers: 是否返回标记图像

        Returns:
            分割后的标记图像，如果return_markers=True则返回(分割图, 标记图)
        """
        if binary_image.ndim != 2:
            raise ValueError("Distance transform watershed requires binary image")

        # 计算距离变换
        distance = ndi.distance_transform_edt(binary_image)

        # 找到局部最大值作为种子点
        from skimage.feature import peak_local_max
        local_max = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=binary_image
        )

        # 创建标记图像
        markers = np.zeros_like(binary_image, dtype=np.int32)
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)

        # 扩展标记
        markers = ndi.label(markers)[0]

        # 执行分水岭
        labels = cv2.watershed(
            cv2.cvtColor((binary_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
            markers
        )

        logger.debug(f"Distance transform watershed: {len(local_max)} seeds, {len(np.unique(labels))} regions")

        if return_markers:
            return labels, markers
        return labels

    @staticmethod
    def watershed_marker_controlled(
        image: np.ndarray,
        binary_mask: np.ndarray,
        sure_fg_erosion: int = 3,
        sure_bg_dilation: int = 3
    ) -> np.ndarray:
        """
        标记控制的分水岭分割

        Args:
            image: 输入图像（灰度图或彩色图）
            binary_mask: 二值掩码（前景为白色）
            sure_fg_erosion: 确定前景的腐蚀核大小
            sure_bg_dilation: 确定背景的膨胀核大小

        Returns:
            分割后的标记图像
        """
        # 确定背景区域（膨胀）
        kernel = np.ones((sure_bg_dilation, sure_bg_dilation), np.uint8)
        sure_bg = cv2.dilate(binary_mask, kernel, iterations=1)

        # 确定前景区域（腐蚀）
        kernel = np.ones((sure_fg_erosion, sure_fg_erosion), np.uint8)
        sure_fg = cv2.erode(binary_mask, kernel, iterations=1)

        # 未知区域
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 标记前景对象
        _, markers = cv2.connectedComponents(sure_fg)

        # 背景标记为1，前景对象从2开始
        markers = markers + 1
        markers[unknown == 255] = 0

        # 执行分水岭
        if image.ndim == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()

        markers = cv2.watershed(image_color, markers)

        logger.debug(f"Marker-controlled watershed: {len(np.unique(markers))} regions")
        return markers

    @staticmethod
    def visualize_watershed(
        image: np.ndarray,
        markers: np.ndarray,
        show_boundaries: bool = True
    ) -> np.ndarray:
        """
        可视化分水岭分割结果

        Args:
            image: 原始图像
            markers: 分水岭标记图像
            show_boundaries: 是否显示边界

        Returns:
            可视化图像
        """
        # 创建彩色标记图像
        if image.ndim == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()

        if show_boundaries:
            # 标记边界为红色
            result[markers == -1] = [0, 0, 255]

        return result


# 便捷函数
def watershed_distance(binary_image: np.ndarray, min_distance: int = 10) -> np.ndarray:
    """基于距离变换的分水岭分割便捷函数"""
    return WatershedSegmentation.watershed_distance_transform(binary_image, min_distance)


def watershed_marker(
    image: np.ndarray,
    binary_mask: np.ndarray,
    erosion: int = 3
) -> np.ndarray:
    """标记控制的分水岭分割便捷函数"""
    return WatershedSegmentation.watershed_marker_controlled(
        image, binary_mask, erosion, erosion
    )

