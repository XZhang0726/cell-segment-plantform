"""
几何辅助细化模块

在高冲突区域使用几何方法（分水岭算法）辅助边界细化
结合DST冲突度信息，实现混合融合策略
"""
import numpy as np
from typing import Optional, Tuple
from scipy.ndimage import gaussian_gradient_magnitude, distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import label
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_gradient_map(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    计算图像梯度图

    Args:
        image: 原始图像 (H, W) 或 (H, W, C)
        sigma: 高斯平滑参数

    Returns:
        梯度图 (H, W)
    """
    # 如果是彩色图像，转换为灰度
    if len(image.shape) == 3:
        # 使用加权平均转换为灰度
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = image.copy()

    # 归一化到0-1范围
    if gray.max() > 1:
        gray = gray.astype(np.float32) / 255.0

    # 计算梯度幅值
    gradient = gaussian_gradient_magnitude(gray, sigma=sigma)

    return gradient


def watershed_refinement(image: np.ndarray,
                         fused_mask: np.ndarray,
                         conflict_map: Optional[np.ndarray] = None,
                         conflict_threshold: float = 0.5,
                         gradient_sigma: float = 1.0) -> Tuple[np.ndarray, dict]:
    """
    使用分水岭算法在高冲突区域细化边界

    这是一个混合融合方法：
    - 低冲突区域：保持DST融合结果
    - 高冲突区域：使用分水岭细化边界

    Args:
        image: 原始图像 (H, W) 或 (H, W, C)
        fused_mask: DST融合后的标签掩码 (H, W)
        conflict_map: 冲突度图 (H, W)，如果为None则对所有区域应用分水岭
        conflict_threshold: 冲突阈值，超过此值的区域使用分水岭
        gradient_sigma: 梯度计算的高斯平滑参数

    Returns:
        (refined_mask, stats): 细化后的掩码和统计信息
    """
    logger.info(f"开始分水岭边界细化，冲突阈值={conflict_threshold}")

    # 1. 计算梯度图作为地形图
    gradient = compute_gradient_map(image, sigma=gradient_sigma)

    # 2. 识别高冲突区域
    if conflict_map is not None:
        high_conflict_mask = conflict_map > conflict_threshold
        high_conflict_pixels = np.sum(high_conflict_mask)
        logger.info(f"高冲突像素数: {high_conflict_pixels} ({high_conflict_pixels / conflict_map.size * 100:.2f}%)")
    else:
        # 如果没有冲突图，对所有前景区域应用分水岭
        high_conflict_mask = fused_mask > 0
        high_conflict_pixels = np.sum(high_conflict_mask)
        logger.info(f"对所有前景区域应用分水岭: {high_conflict_pixels} 像素")

    # 3. 提取确定区域作为种子点（markers）
    # 策略1: 尝试使用低冲突区域作为确定种子
    markers = None
    num_markers = 0

    if conflict_map is not None:
        # 使用更宽松的阈值：conflict_threshold * 0.6
        certain_threshold = conflict_threshold * 0.6
        certain_mask = (conflict_map < certain_threshold) & (fused_mask > 0)
        markers = label(certain_mask)
        num_markers = np.max(markers)
        logger.info(f"策略1（低冲突区域）: 确定种子数={num_markers}, 阈值={certain_threshold:.3f}")

    # 策略2: 如果策略1失败，使用距离变换找到细胞中心作为种子
    if num_markers == 0:
        logger.info("策略1失败，尝试策略2（距离变换）")
        distance = distance_transform_edt(fused_mask > 0)
        # 使用距离变换的局部最大值作为种子
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(distance, size=10)
        certain_mask = (distance == local_max) & (distance > 3)  # 降低距离阈值从5到3
        markers = label(certain_mask)
        num_markers = np.max(markers)
        logger.info(f"策略2（距离变换）: 确定种子数={num_markers}")

    # 策略3: 如果策略2也失败，直接使用融合掩码的标签作为种子
    if num_markers == 0:
        logger.info("策略2失败，尝试策略3（使用融合掩码标签）")
        markers = fused_mask.copy()
        num_markers = np.max(markers)
        logger.info(f"策略3（融合掩码标签）: 确定种子数={num_markers}")

    if num_markers == 0:
        logger.warning("所有策略都失败，没有找到确定种子，返回原始融合结果")
        return fused_mask, {
            'refined': False,
            'reason': 'no_markers',
            'high_conflict_pixels': high_conflict_pixels if conflict_map is not None else 0
        }

    # 4. 在高冲突区域运行分水岭
    # 创建掩码：只在前景区域运行分水岭
    watershed_mask = fused_mask > 0

    try:
        # 运行分水岭算法
        watershed_result = watershed(gradient, markers, mask=watershed_mask)

        # 5. 融合结果：低冲突区域保持不变，高冲突区域使用分水岭
        refined_mask = fused_mask.copy()
        if conflict_map is not None:
            refined_mask[high_conflict_mask] = watershed_result[high_conflict_mask]
        else:
            refined_mask = watershed_result

        # 6. 统计信息
        refined_pixels_mask = refined_mask != fused_mask
        refined_pixels = np.sum(refined_pixels_mask)
        stats = {
            'refined': True,
            'num_markers': num_markers,
            'high_conflict_pixels': high_conflict_pixels if conflict_map is not None else 0,
            'refined_pixels': refined_pixels,
            'refined_percentage': refined_pixels / fused_mask.size * 100,
            'refined_mask': refined_pixels_mask  # 添加细化像素的二值掩码
        }

        logger.info(f"分水岭细化完成：修改了 {refined_pixels} 个像素 ({stats['refined_percentage']:.2f}%)")

        return refined_mask, stats

    except Exception as e:
        logger.error(f"分水岭细化失败: {e}")
        return fused_mask, {
            'refined': False,
            'reason': 'watershed_failed',
            'error': str(e)
        }
