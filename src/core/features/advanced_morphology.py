"""
高级细胞形态学特征提取模块

提供更高级的形态学、纹理和强度特征提取方法
"""
import numpy as np
import pandas as pd
from skimage import measure, feature
from skimage.measure import moments_hu
from scipy import ndimage
from scipy.stats import skew, kurtosis, entropy
from typing import Dict, List, Optional
from loguru import logger


def extract_hu_moments(region) -> Dict[str, float]:
    """
    提取Hu矩特征（7个旋转、缩放、平移不变的形状描述符）

    Args:
        region: skimage regionprops对象

    Returns:
        包含7个Hu矩的字典
    """
    # 获取归一化的中心矩
    hu = moments_hu(region.moments_central)

    # 对Hu矩取对数，使其更适合分析
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return {
        'hu_moment_1': hu_log[0],
        'hu_moment_2': hu_log[1],
        'hu_moment_3': hu_log[2],
        'hu_moment_4': hu_log[3],
        'hu_moment_5': hu_log[4],
        'hu_moment_6': hu_log[5],
        'hu_moment_7': hu_log[6],
    }


def extract_intensity_features(region, image: np.ndarray) -> Dict[str, float]:
    """
    提取强度统计特征

    Args:
        region: skimage regionprops对象
        image: 原始灰度图像

    Returns:
        强度特征字典
    """
    # 获取细胞区域的像素值
    cell_pixels = image[region.coords[:, 0], region.coords[:, 1]]

    # 基础统计
    mean_intensity = np.mean(cell_pixels)
    std_intensity = np.std(cell_pixels)
    min_intensity = np.min(cell_pixels)
    max_intensity = np.max(cell_pixels)
    median_intensity = np.median(cell_pixels)

    # 高级统计
    skewness = skew(cell_pixels)
    kurt = kurtosis(cell_pixels)

    # 计算熵（信息熵）
    hist, _ = np.histogram(cell_pixels, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # 归一化
    ent = entropy(hist + 1e-10)  # 避免log(0)

    # 强度范围和对比度
    intensity_range = max_intensity - min_intensity

    return {
        'intensity_mean': mean_intensity,
        'intensity_std': std_intensity,
        'intensity_min': min_intensity,
        'intensity_max': max_intensity,
        'intensity_median': median_intensity,
        'intensity_range': intensity_range,
        'intensity_skewness': skewness,
        'intensity_kurtosis': kurt,
        'intensity_entropy': ent,
    }


def extract_boundary_features(region, pixel_size: float = 1.0) -> Dict[str, float]:
    """
    提取边界复杂度特征

    Args:
        region: skimage regionprops对象
        pixel_size: 像素大小(μm/pixel)

    Returns:
        边界特征字典
    """
    # 边界粗糙度 = 周长² / (4π * 面积)
    # 完美圆形的粗糙度为1，越粗糙值越大
    roughness = (region.perimeter ** 2) / (4 * np.pi * region.area) if region.area > 0 else 0

    # 紧凑度 = 面积 / 凸包面积
    compactness = region.solidity

    # 凹凸性 = 凸包周长 - 周长
    convex_perimeter = region.perimeter / region.solidity if region.solidity > 0 else region.perimeter
    concavity = (convex_perimeter - region.perimeter) * pixel_size

    # 形状因子 = 4π * 面积 / 周长²（与circularity相同，但这里是标准定义）
    shape_factor = (4 * np.pi * region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0

    return {
        'boundary_roughness': roughness,
        'boundary_compactness': compactness,
        'boundary_concavity': concavity,
        'shape_factor': shape_factor,
    }


def extract_texture_features_glcm(region, image: np.ndarray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]) -> Dict[str, float]:
    """
    提取基于灰度共生矩阵（GLCM）的Haralick纹理特征

    Args:
        region: skimage regionprops对象
        image: 原始灰度图像
        distances: GLCM计算的距离列表
        angles: GLCM计算的角度列表

    Returns:
        纹理特征字典
    """
    # 提取细胞区域的图像
    minr, minc, maxr, maxc = region.bbox
    cell_image = image[minr:maxr, minc:maxc]
    cell_mask = region.image

    # 只保留细胞区域的像素
    cell_image_masked = cell_image * cell_mask

    # 归一化到0-255
    if cell_image_masked.max() > 0:
        cell_image_normalized = ((cell_image_masked - cell_image_masked.min()) /
                                 (cell_image_masked.max() - cell_image_masked.min()) * 255).astype(np.uint8)
    else:
        cell_image_normalized = cell_image_masked.astype(np.uint8)

    try:
        # 计算GLCM
        glcm = feature.graycomatrix(cell_image_normalized, distances=distances, angles=angles,
                                    levels=256, symmetric=True, normed=True)

        # 提取Haralick特征
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        energy = feature.graycoprops(glcm, 'energy').mean()
        correlation = feature.graycoprops(glcm, 'correlation').mean()
        asm = feature.graycoprops(glcm, 'ASM').mean()

        return {
            'texture_contrast': contrast,
            'texture_dissimilarity': dissimilarity,
            'texture_homogeneity': homogeneity,
            'texture_energy': energy,
            'texture_correlation': correlation,
            'texture_asm': asm,
        }
    except Exception as e:
        logger.warning(f"Failed to extract GLCM features: {e}")
        return {
            'texture_contrast': 0,
            'texture_dissimilarity': 0,
            'texture_homogeneity': 0,
            'texture_energy': 0,
            'texture_correlation': 0,
            'texture_asm': 0,
        }


def calculate_fractal_dimension(region) -> float:
    """
    计算边界的分形维数（Box-counting方法）

    Args:
        region: skimage regionprops对象

    Returns:
        分形维数
    """
    try:
        # 获取细胞的二值图像
        binary_image = region.image.astype(bool)

        # 边界提取
        from scipy import ndimage
        boundary = binary_image ^ ndimage.binary_erosion(binary_image)

        # Box-counting算法
        def boxcount(image, k):
            S = np.add.reduceat(
                np.add.reduceat(image, np.arange(0, image.shape[0], k), axis=0),
                np.arange(0, image.shape[1], k), axis=1)
            return len(np.where(S > 0)[0])

        # 计算不同尺度下的box数量
        scales = np.array([2, 4, 8, 16])
        scales = scales[scales < min(boundary.shape)]

        if len(scales) < 2:
            return 0

        counts = []
        for scale in scales:
            counts.append(boxcount(boundary, scale))

        # 线性拟合 log(N) vs log(1/scale)
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        fractal_dim = -coeffs[0]

        return fractal_dim
    except Exception as e:
        logger.warning(f"Failed to calculate fractal dimension: {e}")
        return 0


def extract_advanced_shape_features(region, pixel_size: float = 1.0) -> Dict[str, float]:
    """
    提取高级形状特征

    Args:
        region: skimage regionprops对象
        pixel_size: 像素大小(μm/pixel)

    Returns:
        高级形状特征字典
    """
    # 椭圆度 = 短轴/长轴
    ellipticity = region.minor_axis_length / region.major_axis_length if region.major_axis_length > 0 else 0

    # 伸长度 = 1 - 椭圆度
    elongation = 1 - ellipticity

    # 矩形度 = 面积 / 边界框面积
    bbox_area = (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
    rectangularity = region.area / bbox_area if bbox_area > 0 else 0

    # 等效椭圆面积
    equivalent_ellipse_area = np.pi * region.major_axis_length * region.minor_axis_length / 4

    # 分形维数
    fractal_dim = calculate_fractal_dimension(region)

    return {
        'ellipticity': ellipticity,
        'elongation': elongation,
        'rectangularity': rectangularity,
        'equivalent_ellipse_area': equivalent_ellipse_area * (pixel_size ** 2),
        'fractal_dimension': fractal_dim,
    }


def extract_advanced_cell_features(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    pixel_size: float = 1.0,
    min_area: int = 100,
    include_hu_moments: bool = True,
    include_intensity: bool = True,
    include_texture: bool = True,
    include_boundary: bool = True,
    include_advanced_shape: bool = True
) -> pd.DataFrame:
    """
    提取高级细胞形态学特征（整合所有高级特征）

    Args:
        mask: 分割掩码，每个细胞有唯一标签
        image: 原始灰度图像（用于强度和纹理特征）
        pixel_size: 像素大小(μm/pixel)
        min_area: 最小细胞面积阈值（像素）
        include_hu_moments: 是否包含Hu矩特征
        include_intensity: 是否包含强度特征（需要提供image）
        include_texture: 是否包含纹理特征（需要提供image）
        include_boundary: 是否包含边界特征
        include_advanced_shape: 是否包含高级形状特征

    Returns:
        包含所有高级特征的DataFrame
    """
    # 使用regionprops提取基础信息
    if image is not None:
        regions = measure.regionprops(mask, intensity_image=image)
    else:
        regions = measure.regionprops(mask)

    if len(regions) == 0:
        logger.warning("No cells detected in mask")
        return pd.DataFrame()

    features_list = []
    sequential_id = 0

    for region in regions:
        # 过滤面积过小的细胞
        if region.area < min_area:
            continue

        sequential_id += 1

        # 基础特征（包含所有基础形态学特征）
        features = {
            'sequential_id': sequential_id,
            'cell_id': region.label,
            'centroid_y': region.centroid[0],
            'centroid_x': region.centroid[1],
            'area_pixels': region.area,
            'area_um2': region.area * (pixel_size ** 2),
            'perimeter_pixels': region.perimeter,
            'perimeter_um': region.perimeter * pixel_size,

            # 基础形状特征
            'major_axis_length': region.major_axis_length * pixel_size,
            'minor_axis_length': region.minor_axis_length * pixel_size,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity,
            'extent': region.extent,

            # 计算圆度
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

        # Hu矩特征
        if include_hu_moments:
            try:
                hu_features = extract_hu_moments(region)
                features.update(hu_features)
            except Exception as e:
                logger.warning(f"Failed to extract Hu moments for cell {region.label}: {e}")

        # 强度特征
        if include_intensity and image is not None:
            try:
                intensity_features = extract_intensity_features(region, image)
                features.update(intensity_features)
            except Exception as e:
                logger.warning(f"Failed to extract intensity features for cell {region.label}: {e}")

        # 纹理特征
        if include_texture and image is not None:
            try:
                texture_features = extract_texture_features_glcm(region, image)
                features.update(texture_features)
            except Exception as e:
                logger.warning(f"Failed to extract texture features for cell {region.label}: {e}")

        # 边界特征
        if include_boundary:
            try:
                boundary_features = extract_boundary_features(region, pixel_size)
                features.update(boundary_features)
            except Exception as e:
                logger.warning(f"Failed to extract boundary features for cell {region.label}: {e}")

        # 高级形状特征
        if include_advanced_shape:
            try:
                shape_features = extract_advanced_shape_features(region, pixel_size)
                features.update(shape_features)
            except Exception as e:
                logger.warning(f"Failed to extract advanced shape features for cell {region.label}: {e}")

        features_list.append(features)

    df = pd.DataFrame(features_list)
    logger.info(f"Extracted advanced features for {len(df)} cells")

    return df
