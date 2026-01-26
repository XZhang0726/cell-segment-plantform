"""
置信度工具模块

为没有置信度输出的分割模型生成合成置信度图
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import List
from ..utils.logger import get_logger

logger = get_logger(__name__)


def generate_confidence_from_mask(mask: np.ndarray,
                                  base_confidence: float = 0.8,
                                  boundary_penalty: float = 0.3) -> np.ndarray:
    """
    从分割掩码生成合成置信度图

    策略：基于距离变换，边界附近置信度低，中心区域置信度高
    这模拟了真实模型的行为：边界区域更不确定

    Args:
        mask: 标签掩码 (H, W)，值为实例ID
        base_confidence: 基础置信度 [0,1]
        boundary_penalty: 边界惩罚系数 [0,1]

    Returns:
        置信度图 (H, W)，值范围 [0,1]
    """
    H, W = mask.shape
    confidence_map = np.zeros((H, W), dtype=np.float32)

    # 获取所有实例ID
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]  # 排除背景

    for inst_id in instance_ids:
        # 提取该实例的二值掩码
        binary_mask = (mask == inst_id).astype(np.uint8)

        # 计算距离变换（到边界的距离）
        distance = distance_transform_edt(binary_mask)

        # 归一化距离到 [0, 1]
        if distance.max() > 0:
            normalized_distance = distance / distance.max()
        else:
            normalized_distance = np.zeros_like(distance)

        # 计算置信度：中心高，边界低
        # confidence = base - penalty * (1 - normalized_distance)
        instance_confidence = base_confidence - boundary_penalty * (1 - normalized_distance)
        instance_confidence = np.clip(instance_confidence, 0.0, 1.0)

        # 写入置信度图
        confidence_map[binary_mask > 0] = instance_confidence[binary_mask > 0]

    return confidence_map


def generate_confidence_maps(masks_list: List[np.ndarray],
                            model_names: List[str],
                            model_reliabilities: dict) -> List[np.ndarray]:
    """
    为多个模型生成置信度图

    根据模型可靠性调整基础置信度：
    - 高可靠性模型（如CellViT）→ 高基础置信度
    - 低可靠性模型（如传统方法）→ 低基础置信度

    Args:
        masks_list: 掩码列表
        model_names: 模型名称列表
        model_reliabilities: 模型可靠性字典

    Returns:
        置信度图列表
    """
    logger.info(f"生成 {len(masks_list)} 个模型的合成置信度图...")

    confidences_list = []

    for mask, model_name in zip(masks_list, model_names):
        # 根据模型可靠性调整基础置信度
        reliability = model_reliabilities.get(model_name, 0.8)
        base_confidence = reliability * 0.9  # 略低于可靠性

        # 生成置信度图
        confidence_map = generate_confidence_from_mask(
            mask,
            base_confidence=base_confidence,
            boundary_penalty=0.3
        )

        confidences_list.append(confidence_map)

        logger.debug(f"  {model_name}: base_confidence={base_confidence:.2f}, "
                    f"mean={np.mean(confidence_map[mask > 0]):.3f}")

    logger.info("置信度图生成完成")

    return confidences_list
