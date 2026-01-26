"""
不确定性计算模块

计算模型间的分歧热图和一致性指标
"""
import numpy as np
from typing import List, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_disagreement_map(masks_list: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    计算分歧图（Disagreement Map）

    Args:
        masks_list: 多个模型的标签掩码列表

    Returns:
        disagreement_map: 不确定性热图 (0-1范围)
        consistency_score: 全局一致性分数
    """
    logger.info(f"计算分歧图，共{len(masks_list)}个模型")

    # 1. 将标签掩码转换为二值掩码（前景/背景）
    binary_masks = []
    for mask in masks_list:
        binary = (mask > 0).astype(np.float32)
        binary_masks.append(binary)

    # 2. 堆叠所有二值掩码
    stack = np.stack(binary_masks, axis=0)

    # 3. 计算每个像素的投票比例
    vote_ratio = np.mean(stack, axis=0)  # 范围 [0, 1]

    # 4. 计算分歧度
    # 当vote_ratio=0.5时，分歧度最大（模型完全分裂）
    # 当vote_ratio=0或1时，分歧度最小（模型完全一致）
    disagreement = 1.0 - np.abs(2 * vote_ratio - 1)

    # 5. 计算全局一致性分数
    consistency_score = 1.0 - np.mean(disagreement)

    logger.info(f"全局一致性分数: {consistency_score:.4f}")

    return disagreement, consistency_score


def compute_model_consistency(masks_list: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    计算模型间的成对一致性

    Args:
        masks_list: 多个模型的标签掩码列表

    Returns:
        consistency_matrix: 模型间的IoU矩阵
        avg_consistency: 平均一致性
    """
    logger.info(f"计算模型间一致性，共{len(masks_list)}个模型")

    n_models = len(masks_list)
    consistency_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(i, n_models):
            if i == j:
                consistency_matrix[i, j] = 1.0
            else:
                # 计算两个模型的整体IoU
                mask_i = (masks_list[i] > 0)
                mask_j = (masks_list[j] > 0)
                intersection = np.logical_and(mask_i, mask_j)
                union = np.logical_or(mask_i, mask_j)
                iou = np.sum(intersection) / (np.sum(union) + 1e-8)
                consistency_matrix[i, j] = iou
                consistency_matrix[j, i] = iou

    # 计算平均一致性（排除对角线）
    mask = ~np.eye(n_models, dtype=bool)
    avg_consistency = np.mean(consistency_matrix[mask])

    logger.info(f"平均模型一致性: {avg_consistency:.4f}")

    return consistency_matrix, avg_consistency
