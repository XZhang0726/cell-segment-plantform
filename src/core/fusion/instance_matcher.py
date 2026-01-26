"""
实例匹配模块

使用IoU阈值和贪婪匹配算法找出不同模型中代表同一个细胞的预测
"""
import numpy as np
from typing import List, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算两个二值掩码的IoU (Intersection over Union)

    Args:
        mask1: 第一个二值掩码
        mask2: 第二个二值掩码

    Returns:
        IoU值 (0-1范围)
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    union_sum = np.sum(union)
    if union_sum == 0:
        return 0.0

    return np.sum(intersection) / union_sum


def extract_instances(mask: np.ndarray, model_idx: int) -> List[dict]:
    """
    从标签掩码中提取所有实例

    Args:
        mask: 标签掩码，每个细胞有唯一的整数ID
        model_idx: 模型索引

    Returns:
        实例列表，每个实例包含 {model_idx, instance_id, binary_mask, area}
    """
    instances = []
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]  # 跳过背景

    for inst_id in unique_ids:
        binary_mask = (mask == inst_id)
        area = np.sum(binary_mask)

        instances.append({
            'model_idx': model_idx,
            'instance_id': int(inst_id),
            'binary_mask': binary_mask,
            'area': int(area)
        })

    return instances


def match_instances(masks_list: List[np.ndarray],
                   iou_threshold: float = 0.5) -> List[List[Tuple[int, int]]]:
    """
    匹配多个模型的实例

    使用贪婪匹配算法：
    1. 从所有模型中提取实例，构建候选池
    2. 按面积排序（大细胞优先）
    3. 对每个实例，找出与其IoU>threshold的其他实例
    4. 将匹配的实例分组

    Args:
        masks_list: 多个模型的标签掩码列表
        iou_threshold: IoU匹配阈值

    Returns:
        匹配组列表，每组包含 [(model_idx, instance_id), ...]
    """
    logger.info(f"开始实例匹配，共{len(masks_list)}个模型，IoU阈值={iou_threshold}")

    # 1. 提取所有模型的所有实例，构建候选池
    proposals = []
    for model_idx, mask in enumerate(masks_list):
        instances = extract_instances(mask, model_idx)
        proposals.extend(instances)
        logger.info(f"模型{model_idx}提取到{len(instances)}个实例")

    logger.info(f"候选池共{len(proposals)}个实例")

    # 2. 按面积排序（大细胞优先，通常更鲁棒）
    sorted_indices = np.argsort([-p['area'] for p in proposals])

    # 3. 贪婪匹配
    matched_groups = []
    processed_indices = set()

    for i in sorted_indices:
        if i in processed_indices:
            continue

        # 当前实例作为组的起点
        current_proposal = proposals[i]
        current_group = [(current_proposal['model_idx'], current_proposal['instance_id'])]
        processed_indices.add(i)

        # 在剩余的实例中寻找与当前实例IoU足够高的
        for j in sorted_indices:
            if j in processed_indices:
                continue

            other_proposal = proposals[j]

            # 跳过来自同一模型的实例（一个模型不能匹配自己）
            if other_proposal['model_idx'] == current_proposal['model_idx']:
                continue

            # 计算IoU
            iou = compute_iou(current_proposal['binary_mask'],
                            other_proposal['binary_mask'])

            if iou > iou_threshold:
                current_group.append((other_proposal['model_idx'],
                                    other_proposal['instance_id']))
                processed_indices.add(j)

        # 将匹配组添加到结果中
        matched_groups.append(current_group)

    logger.info(f"匹配完成，共{len(matched_groups)}个实例组")

    # 统计匹配情况
    multi_model_groups = [g for g in matched_groups if len(g) > 1]
    logger.info(f"其中{len(multi_model_groups)}个组包含多个模型的预测")

    return matched_groups
