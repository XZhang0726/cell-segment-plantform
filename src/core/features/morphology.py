"""
细胞形态学特征提取模块

提取单个细胞的几何形态学特征
"""
import numpy as np
import pandas as pd
from skimage import measure
from typing import Dict, List
from loguru import logger


def extract_cell_features(mask: np.ndarray, pixel_size: float = 1.0, min_area: int = 100) -> pd.DataFrame:
    """
    提取细胞形态学特征

    Args:
        mask: 分割掩码，每个细胞有唯一标签
        pixel_size: 像素大小(μm/pixel)，用于转换为实际尺寸
        min_area: 最小细胞面积阈值（像素），过滤掉面积小于此值的细胞

    Returns:
        包含所有细胞特征的DataFrame
    """
    # 使用regionprops提取特征
    regions = measure.regionprops(mask)

    if len(regions) == 0:
        logger.warning("No cells detected in mask")
        return pd.DataFrame()

    features_list = []
    sequential_id = 0  # 连续ID计数器

    for region in regions:
        # 过滤面积过小的细胞
        if region.area < min_area:
            continue

        sequential_id += 1  # 递增连续ID

        features = {
            # 基本标识 - 双ID系统
            'sequential_id': sequential_id,  # 连续ID (1, 2, 3...)
            'cell_id': region.label,  # 原始mask标签ID（可能不连续）

            # 位置特征
            'centroid_y': region.centroid[0],
            'centroid_x': region.centroid[1],

            # 面积和周长特征
            'area_pixels': region.area,
            'area_um2': region.area * (pixel_size ** 2),
            'perimeter_pixels': region.perimeter,
            'perimeter_um': region.perimeter * pixel_size,

            # 形状特征
            'major_axis_length': region.major_axis_length * pixel_size,
            'minor_axis_length': region.minor_axis_length * pixel_size,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity,
            'extent': region.extent,

            # 计算圆度 (4π*面积/周长²)
            'circularity': (4 * np.pi * region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0,

            # 长宽比
            'aspect_ratio': region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0,

            # 等效直径
            'equivalent_diameter_pixels': region.equivalent_diameter,
            'equivalent_diameter_um': region.equivalent_diameter * pixel_size,

            # 边界框
            'bbox_min_row': region.bbox[0],
            'bbox_min_col': region.bbox[1],
            'bbox_max_row': region.bbox[2],
            'bbox_max_col': region.bbox[3],
        }

        features_list.append(features)

    df = pd.DataFrame(features_list)

    logger.info(f"Extracted features for {len(df)} cells")

    return df


def get_feature_statistics(df: pd.DataFrame) -> Dict:
    """
    计算特征的统计信息

    Args:
        df: 特征DataFrame

    Returns:
        统计信息字典
    """
    if df.empty:
        return {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['cell_id', 'centroid_x', 'centroid_y',
                                                                 'bbox_min_row', 'bbox_min_col',
                                                                 'bbox_max_row', 'bbox_max_col']]

    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }

    return stats


def filter_cells_by_features(df: pd.DataFrame,
                             min_area: float = None,
                             max_area: float = None,
                             min_circularity: float = None,
                             max_circularity: float = None) -> pd.DataFrame:
    """
    根据特征过滤细胞

    Args:
        df: 特征DataFrame
        min_area: 最小面积(μm²)
        max_area: 最大面积(μm²)
        min_circularity: 最小圆度
        max_circularity: 最大圆度

    Returns:
        过滤后的DataFrame
    """
    filtered_df = df.copy()

    if min_area is not None:
        filtered_df = filtered_df[filtered_df['area_um2'] >= min_area]

    if max_area is not None:
        filtered_df = filtered_df[filtered_df['area_um2'] <= max_area]

    if min_circularity is not None:
        filtered_df = filtered_df[filtered_df['circularity'] >= min_circularity]

    if max_circularity is not None:
        filtered_df = filtered_df[filtered_df['circularity'] <= max_circularity]

    logger.info(f"Filtered from {len(df)} to {len(filtered_df)} cells")

    return filtered_df
