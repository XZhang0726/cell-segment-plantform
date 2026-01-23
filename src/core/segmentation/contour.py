"""
轮廓检测和分析模块

提供轮廓检测、特征提取和分析方法
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ContourAnalysis:
    """轮廓分析类"""

    @staticmethod
    def find_contours(
        binary_image: np.ndarray,
        mode: str = 'external',
        method: str = 'simple'
    ) -> List[np.ndarray]:
        """
        查找轮廓

        Args:
            binary_image: 二值图像
            mode: 轮廓检索模式 ('external', 'list', 'tree', 'ccomp')
            method: 轮廓近似方法 ('none', 'simple', 'tc89_l1', 'tc89_kcos')

        Returns:
            轮廓列表
        """
        if binary_image.ndim != 2:
            raise ValueError("Contour detection requires binary image")

        # 轮廓检索模式
        modes = {
            'external': cv2.RETR_EXTERNAL,
            'list': cv2.RETR_LIST,
            'tree': cv2.RETR_TREE,
            'ccomp': cv2.RETR_CCOMP
        }

        # 轮廓近似方法
        methods = {
            'none': cv2.CHAIN_APPROX_NONE,
            'simple': cv2.CHAIN_APPROX_SIMPLE,
            'tc89_l1': cv2.CHAIN_APPROX_TC89_L1,
            'tc89_kcos': cv2.CHAIN_APPROX_TC89_KCOS
        }

        if mode not in modes:
            raise ValueError(f"Unknown contour mode: {mode}")
        if method not in methods:
            raise ValueError(f"Unknown contour method: {method}")

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary_image,
            modes[mode],
            methods[method]
        )

        logger.debug(f"Found {len(contours)} contours")
        return contours

    @staticmethod
    def filter_contours(
        contours: List[np.ndarray],
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
        min_perimeter: Optional[float] = None,
        max_perimeter: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        根据面积和周长过滤轮廓

        Args:
            contours: 轮廓列表
            min_area: 最小面积
            max_area: 最大面积
            min_perimeter: 最小周长
            max_perimeter: 最大周长

        Returns:
            过滤后的轮廓列表
        """
        filtered = []

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # 检查面积条件
            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            # 检查周长条件
            if min_perimeter is not None and perimeter < min_perimeter:
                continue
            if max_perimeter is not None and perimeter > max_perimeter:
                continue

            filtered.append(contour)

        logger.debug(f"Filtered {len(contours)} -> {len(filtered)} contours")
        return filtered

    @staticmethod
    def get_contour_properties(contour: np.ndarray) -> Dict[str, float]:
        """
        获取轮廓的各种属性

        Args:
            contour: 单个轮廓

        Returns:
            包含轮廓属性的字典
        """
        properties = {}

        # 基本属性
        properties['area'] = cv2.contourArea(contour)
        properties['perimeter'] = cv2.arcLength(contour, True)

        # 圆形度 (4π*area/perimeter^2)
        if properties['perimeter'] > 0:
            properties['circularity'] = (4 * np.pi * properties['area']) / (properties['perimeter'] ** 2)
        else:
            properties['circularity'] = 0

        # 边界框
        x, y, w, h = cv2.boundingRect(contour)
        properties['bbox_x'] = x
        properties['bbox_y'] = y
        properties['bbox_width'] = w
        properties['bbox_height'] = h
        properties['bbox_area'] = w * h

        # 长宽比
        if h > 0:
            properties['aspect_ratio'] = w / h
        else:
            properties['aspect_ratio'] = 0

        # 矩形度 (轮廓面积/边界框面积)
        if properties['bbox_area'] > 0:
            properties['extent'] = properties['area'] / properties['bbox_area']
        else:
            properties['extent'] = 0

        # 凸包
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        properties['convex_hull_area'] = hull_area

        # 凸度 (轮廓面积/凸包面积)
        if hull_area > 0:
            properties['solidity'] = properties['area'] / hull_area
        else:
            properties['solidity'] = 0

        # 等效直径
        properties['equivalent_diameter'] = np.sqrt(4 * properties['area'] / np.pi)

        return properties

    @staticmethod
    def draw_contours(
        image: np.ndarray,
        contours: List[np.ndarray],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        fill: bool = False
    ) -> np.ndarray:
        """
        在图像上绘制轮廓

        Args:
            image: 输入图像
            contours: 轮廓列表
            color: 轮廓颜色 (B, G, R)
            thickness: 线条粗细，-1表示填充
            fill: 是否填充轮廓

        Returns:
            绘制了轮廓的图像
        """
        # 创建副本以避免修改原图
        if image.ndim == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()

        # 绘制轮廓
        thickness_value = -1 if fill else thickness
        cv2.drawContours(result, contours, -1, color, thickness_value)

        logger.debug(f"Drew {len(contours)} contours")
        return result

    @staticmethod
    def get_bounding_boxes(
        contours: List[np.ndarray],
        rotated: bool = False
    ) -> List[Tuple]:
        """
        获取轮廓的边界框

        Args:
            contours: 轮廓列表
            rotated: 是否使用旋转边界框

        Returns:
            边界框列表
        """
        boxes = []

        for contour in contours:
            if rotated:
                # 旋转边界框 (中心点, 尺寸, 角度)
                if len(contour) >= 5:
                    box = cv2.minAreaRect(contour)
                    boxes.append(box)
            else:
                # 直立边界框 (x, y, w, h)
                box = cv2.boundingRect(contour)
                boxes.append(box)

        logger.debug(f"Got {len(boxes)} bounding boxes")
        return boxes

    @staticmethod
    def fit_ellipse(
        contours: List[np.ndarray],
        min_points: int = 5
    ) -> List[Optional[Tuple]]:
        """
        为轮廓拟合椭圆

        Args:
            contours: 轮廓列表
            min_points: 拟合椭圆所需的最小点数

        Returns:
            椭圆参数列表 (中心点, 轴长, 角度)
        """
        ellipses = []

        for contour in contours:
            if len(contour) >= min_points:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    ellipses.append(ellipse)
                except:
                    ellipses.append(None)
            else:
                ellipses.append(None)

        logger.debug(f"Fitted {sum(e is not None for e in ellipses)} ellipses")
        return ellipses


# 便捷函数
def find_contours(binary_image: np.ndarray, mode: str = 'external') -> List[np.ndarray]:
    """查找轮廓的便捷函数"""
    return ContourAnalysis.find_contours(binary_image, mode)


def filter_by_area(
    contours: List[np.ndarray],
    min_area: float = 100,
    max_area: Optional[float] = None
) -> List[np.ndarray]:
    """根据面积过滤轮廓的便捷函数"""
    return ContourAnalysis.filter_contours(contours, min_area=min_area, max_area=max_area)


def get_properties(contour: np.ndarray) -> Dict[str, float]:
    """获取轮廓属性的便捷函数"""
    return ContourAnalysis.get_contour_properties(contour)


def draw_contours(
    image: np.ndarray,
    contours: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """绘制轮廓的便捷函数"""
    return ContourAnalysis.draw_contours(image, contours, color)
