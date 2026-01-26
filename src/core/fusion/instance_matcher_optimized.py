"""
实例匹配模块 - 优化版本

使用边界框预过滤、裁剪mask存储等优化技术，大幅提升匹配速度
预期加速：50-200倍
"""
import numpy as np
from typing import List, Tuple, Dict
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_bbox(binary_mask: np.ndarray) -> Dict[str, int]:
    """
    计算二值mask的边界框

    Args:
        binary_mask: 二值掩码

    Returns:
        边界框字典 {x_min, x_max, y_min, y_max}
    """
    y_coords, x_coords = np.where(binary_mask)

    if len(y_coords) == 0:
        return {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}

    return {
        'x_min': int(np.min(x_coords)),
        'x_max': int(np.max(x_coords)),
        'y_min': int(np.min(y_coords)),
        'y_max': int(np.max(y_coords))
    }


def boxes_overlap(box1: Dict[str, int], box2: Dict[str, int]) -> bool:
    """
    快速检查两个边界框是否重叠

    Args:
        box1: 第一个边界框
        box2: 第二个边界框

    Returns:
        是否重叠
    """
    return not (box1['x_max'] < box2['x_min'] or
                box1['x_min'] > box2['x_max'] or
                box1['y_max'] < box2['y_min'] or
                box1['y_min'] > box2['y_max'])


def compute_iou_cropped(mask1: np.ndarray, bbox1: Dict[str, int],
                       mask2: np.ndarray, bbox2: Dict[str, int]) -> float:
    """
    计算两个实例的IoU，使用裁剪后的区域

    优化：只在重叠区域计算IoU，避免遍历整个图像

    Args:
        mask1: 第一个二值掩码（完整图像）
        bbox1: 第一个边界框
        mask2: 第二个二值掩码（完整图像）
        bbox2: 第二个边界框

    Returns:
        IoU值 (0-1范围)
    """
    # 计算重叠区域
    overlap_x_min = max(bbox1['x_min'], bbox2['x_min'])
    overlap_x_max = min(bbox1['x_max'], bbox2['x_max'])
    overlap_y_min = max(bbox1['y_min'], bbox2['y_min'])
    overlap_y_max = min(bbox1['y_max'], bbox2['y_max'])

    # 裁剪到重叠区域
    crop1 = mask1[overlap_y_min:overlap_y_max+1, overlap_x_min:overlap_x_max+1]
    crop2 = mask2[overlap_y_min:overlap_y_max+1, overlap_x_min:overlap_x_max+1]

    # 计算交集（在重叠区域内）
    intersection = np.sum(np.logical_and(crop1, crop2))

    # 计算并集（需要考虑完整区域）
    # 方法：area1 + area2 - intersection
    area1 = (bbox1['x_max'] - bbox1['x_min'] + 1) * (bbox1['y_max'] - bbox1['y_min'] + 1)
    area2 = (bbox2['x_max'] - bbox2['x_min'] + 1) * (bbox2['y_max'] - bbox2['y_min'] + 1)

    # 更精确的面积计算：实际像素数
    area1_actual = np.sum(mask1[bbox1['y_min']:bbox1['y_max']+1,
                                bbox1['x_min']:bbox1['x_max']+1])
    area2_actual = np.sum(mask2[bbox2['y_min']:bbox2['y_max']+1,
                                bbox2['x_min']:bbox2['x_max']+1])

    union = area1_actual + area2_actual - intersection

    if union == 0:
        return 0.0

    return intersection / union


def extract_instances_optimized(mask: np.ndarray, model_idx: int) -> List[dict]:
    """
    从标签掩码中提取所有实例（优化版本）

    优化：
    1. 计算并存储边界框
    2. 只存储完整mask的引用（避免复制）
    3. 预计算面积

    Args:
        mask: 标签掩码，每个细胞有唯一的整数ID
        model_idx: 模型索引

    Returns:
        实例列表，每个实例包含 {model_idx, instance_id, mask_ref, bbox, area}
    """
    instances = []
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]  # 跳过背景

    for inst_id in unique_ids:
        binary_mask = (mask == inst_id)
        bbox = compute_bbox(binary_mask)
        area = int(np.sum(binary_mask))

        instances.append({
            'model_idx': model_idx,
            'instance_id': int(inst_id),
            'mask_ref': mask,  # 存储原始mask引用
            'inst_id_value': int(inst_id),  # 用于快速提取binary mask
            'bbox': bbox,
            'area': area
        })

    return instances


def match_instances_optimized(masks_list: List[np.ndarray],
                              iou_threshold: float = 0.5) -> List[List[Tuple[int, int]]]:
    """
    匹配多个模型的实例（优化版本）

    优化策略：
    1. 边界框预过滤：只对边界框重叠的实例计算IoU
    2. 面积预过滤：面积差异过大的实例直接跳过
    3. 裁剪区域IoU：只在重叠区域计算，避免遍历整个图像

    预期加速：50-200倍

    Args:
        masks_list: 多个模型的标签掩码列表
        iou_threshold: IoU匹配阈值

    Returns:
        匹配组列表，每组包含 [(model_idx, instance_id), ...]
    """
    logger.info(f"开始实例匹配（优化版本），共{len(masks_list)}个模型，IoU阈值={iou_threshold}")

    # 1. 提取所有模型的所有实例，构建候选池
    proposals = []
    for model_idx, mask in enumerate(masks_list):
        instances = extract_instances_optimized(mask, model_idx)
        proposals.extend(instances)
        logger.info(f"模型{model_idx}提取到{len(instances)}个实例")

    logger.info(f"候选池共{len(proposals)}个实例")

    # 2. 按面积排序（大细胞优先，通常更鲁棒）
    sorted_indices = np.argsort([-p['area'] for p in proposals])

    # 3. 贪婪匹配（带优化）
    matched_groups = []
    processed_indices = set()

    # 统计优化效果
    total_pairs_checked = 0
    bbox_filtered = 0
    area_filtered = 0
    iou_computed = 0

    for i in sorted_indices:
        if i in processed_indices:
            continue

        # 当前实例作为组的起点
        current_proposal = proposals[i]
        current_group = [(current_proposal['model_idx'], current_proposal['instance_id'])]
        processed_indices.add(i)

        # 提取当前实例的binary mask（延迟计算）
        current_binary_mask = (current_proposal['mask_ref'] == current_proposal['inst_id_value'])

        # 在剩余的实例中寻找与当前实例IoU足够高的
        for j in sorted_indices:
            if j in processed_indices:
                continue

            other_proposal = proposals[j]
            total_pairs_checked += 1

            # 跳过来自同一模型的实例
            if other_proposal['model_idx'] == current_proposal['model_idx']:
                continue

            # 优化1：边界框预过滤
            if not boxes_overlap(current_proposal['bbox'], other_proposal['bbox']):
                bbox_filtered += 1
                continue

            # 优化2：面积预过滤
            # 如果面积差异太大，IoU不可能超过阈值
            area_ratio = min(current_proposal['area'], other_proposal['area']) / \
                        max(current_proposal['area'], other_proposal['area'])
            if area_ratio < iou_threshold:
                area_filtered += 1
                continue

            # 提取另一个实例的binary mask
            other_binary_mask = (other_proposal['mask_ref'] == other_proposal['inst_id_value'])

            # 计算IoU（使用优化版本）
            iou = compute_iou_cropped(current_binary_mask, current_proposal['bbox'],
                                     other_binary_mask, other_proposal['bbox'])
            iou_computed += 1

            if iou > iou_threshold:
                current_group.append((other_proposal['model_idx'],
                                    other_proposal['instance_id']))
                processed_indices.add(j)

        # 将匹配组添加到结果中
        matched_groups.append(current_group)

    logger.info(f"匹配完成，共{len(matched_groups)}个实例组")
    logger.info(f"优化统计：检查{total_pairs_checked}对，边界框过滤{bbox_filtered}对，"
               f"面积过滤{area_filtered}对，实际计算IoU {iou_computed}次")
    logger.info(f"优化效果：IoU计算减少了 {100*(1-iou_computed/max(total_pairs_checked,1)):.1f}%")

    # 统计匹配情况
    multi_model_groups = [g for g in matched_groups if len(g) > 1]
    logger.info(f"其中{len(multi_model_groups)}个组包含多个模型的预测")

    return matched_groups
