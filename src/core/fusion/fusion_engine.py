"""
融合引擎模块

对匹配好的实例进行像素级融合，支持多种融合策略
包括简单策略（投票、并集、交集）和高级策略（Dempster-Shafer理论）
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from ..utils.logger import get_logger
from .dempster_shafer import DempsterShaferFusion, FusionResult, handle_conflict

logger = get_logger(__name__)


def _majority_voting(binary_masks: List[np.ndarray]) -> np.ndarray:
    """
    简单多数投票：超过50%的模型同意

    Args:
        binary_masks: 二值掩码列表

    Returns:
        融合后的二值掩码
    """
    stack = np.stack(binary_masks, axis=0)
    vote_count = np.sum(stack, axis=0)
    return vote_count >= (len(binary_masks) / 2)


def _weighted_voting(binary_masks: List[np.ndarray],
                    weights: Optional[List[float]],
                    group: List[Tuple[int, int]]) -> np.ndarray:
    """
    加权投票：考虑模型权重

    Args:
        binary_masks: 二值掩码列表
        weights: 模型权重列表
        group: 匹配组，包含(model_idx, instance_id)元组

    Returns:
        融合后的二值掩码
    """
    if weights is None:
        weights = [1.0] * len(binary_masks)

    weighted_sum = np.zeros_like(binary_masks[0], dtype=np.float32)
    total_weight = 0.0

    for i, (model_idx, _) in enumerate(group):
        weighted_sum += binary_masks[i] * weights[model_idx]
        total_weight += weights[model_idx]

    return (weighted_sum / total_weight) >= 0.5


def _union_fusion(binary_masks: List[np.ndarray]) -> np.ndarray:
    """
    取并集：激进策略，召回率高

    Args:
        binary_masks: 二值掩码列表

    Returns:
        融合后的二值掩码
    """
    stack = np.stack(binary_masks, axis=0)
    return np.any(stack, axis=0)


def _intersection_fusion(binary_masks: List[np.ndarray]) -> np.ndarray:
    """
    取交集：保守策略，精确率高

    Args:
        binary_masks: 二值掩码列表

    Returns:
        融合后的二值掩码
    """
    stack = np.stack(binary_masks, axis=0)
    return np.all(stack, axis=0)


def fuse_instances(matched_groups: List[List[Tuple[int, int]]],
                  masks_list: List[np.ndarray],
                  strategy: str = 'majority',
                  weights: Optional[List[float]] = None,
                  min_vote_count: int = 2) -> np.ndarray:
    """
    融合匹配好的实例组

    Args:
        matched_groups: 匹配组列表，每组包含(model_idx, instance_id)元组
        masks_list: 原始标签掩码列表
        strategy: 融合策略 ('majority', 'weighted', 'union', 'intersection')
        weights: 模型权重（用于weighted策略）
        min_vote_count: 最小投票数

    Returns:
        融合后的标签掩码
    """
    logger.info(f"开始融合，策略={strategy}，最小投票数={min_vote_count}")

    # 1. 初始化输出掩码
    fused_mask = np.zeros_like(masks_list[0], dtype=np.int32)
    instance_id = 1

    # 统计信息
    total_groups = len(matched_groups)
    fused_count = 0
    skipped_count = 0

    # 2. 遍历每个匹配组
    for group in matched_groups:
        # 检查投票数是否满足要求
        contributing_models = set([model_idx for model_idx, _ in group])
        if len(contributing_models) < min_vote_count:
            skipped_count += 1
            continue

        # 3. 提取该组所有实例的二值掩码
        binary_masks = []
        for model_idx, inst_id in group:
            mask = (masks_list[model_idx] == inst_id)
            binary_masks.append(mask)

        # 4. 根据策略进行融合
        if strategy == 'majority':
            consensus_mask = _majority_voting(binary_masks)
        elif strategy == 'weighted':
            consensus_mask = _weighted_voting(binary_masks, weights, group)
        elif strategy == 'union':
            consensus_mask = _union_fusion(binary_masks)
        elif strategy == 'intersection':
            consensus_mask = _intersection_fusion(binary_masks)
        else:
            logger.warning(f"未知的融合策略: {strategy}，使用majority")
            consensus_mask = _majority_voting(binary_masks)

        # 5. 写入融合掩码
        if np.sum(consensus_mask) > 0:
            fused_mask[consensus_mask] = instance_id
            instance_id += 1
            fused_count += 1

    logger.info(f"融合完成：总组数={total_groups}，融合={fused_count}，跳过={skipped_count}")

    return fused_mask


def fuse_instances_dst(matched_groups: List[List[Tuple[int, int]]],
                       masks_list: List[np.ndarray],
                       confidences_list: List[Optional[np.ndarray]],
                       model_names: List[str],
                       model_reliabilities: Dict[str, float],
                       image: Optional[np.ndarray] = None,
                       min_vote_count: int = 2,
                       conflict_threshold: float = 0.6,
                       enable_watershed: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    使用Dempster-Shafer理论进行高级融合

    这是一个高级融合方法，相比简单投票策略，DST能够：
    1. 明确建模不确定性
    2. 量化模型之间的冲突
    3. 提供更严格的数学框架
    4. 输出信念区间和冲突图

    Args:
        matched_groups: 匹配组列表，每组包含(model_idx, instance_id)元组
        masks_list: 原始标签掩码列表
        confidences_list: 每个模型的置信度图列表（可以为None）
        model_names: 模型名称列表
        model_reliabilities: 模型可靠性字典，例如 {'cellpose': 0.9, 'cellvit': 0.85}
        image: 原始图像 (H, W) 或 (H, W, C)，用于分水岭边界细化（可选）
        min_vote_count: 最小投票数
        conflict_threshold: 冲突阈值，超过此值的实例将被标记
        enable_watershed: 是否启用分水岭边界细化（默认True）

    Returns:
        (融合后的标签掩码, DST统计信息字典)

        统计信息包含:
        - 'conflict_map': 冲突热力图
        - 'uncertainty_map': 不确定性热力图
        - 'high_conflict_instances': 高冲突实例列表
        - 'fusion_results': 每个实例的详细融合结果
        - 'watershed_refinement': 分水岭细化统计信息（如果启用）
    """
    logger.info(f"开始DST高级融合，最小投票数={min_vote_count}，冲突阈值={conflict_threshold}")

    # 1. 初始化DST融合引擎
    dst_engine = DempsterShaferFusion(model_reliabilities)

    # 2. 初始化输出
    fused_mask = np.zeros_like(masks_list[0], dtype=np.int32)
    instance_id = 1

    # 3. 统计信息
    total_groups = len(matched_groups)
    fused_count = 0
    skipped_count = 0
    high_conflict_count = 0

    # 4. 策略使用统计
    strategy_counts = {
        'ULTRA_AGGRESSIVE': 0,      # 超激进：任意1个模型（conf>0.75, conflict<0.2）
        'AGGRESSIVE': 0,             # 激进：30%阈值（conf>0.75, conflict<0.4）
        'RELAXED': 0,                # 宽松：40%阈值（conf>0.55, conflict<0.3）
        'STANDARD_HIGH_CONF': 0,     # 标准-高置信度：50%阈值
        'STANDARD_MID_CONF': 0,      # 标准-中置信度：50%阈值
        'STANDARD_LOW_CONF': 0,      # 标准-低置信度：50%阈值
        'STRICT': 0,                 # 严格：60%阈值（conf>0.55, conflict>0.5）
        'STRICT_LOW_CONF': 0,        # 严格-低置信度：60%阈值
        'ULTRA_STRICT': 0,           # 超严格：70%阈值（conf≤0.55, conflict>0.6）
        'INTERSECTION': 0            # 交集：全部同意
    }

    # 5. 存储每个实例的融合结果
    fusion_results = []
    high_conflict_instances = []

    # 5. 遍历每个匹配组
    for group_idx, group in enumerate(matched_groups):
        # 检查投票数
        contributing_models = set([model_idx for model_idx, _ in group])
        if len(contributing_models) < min_vote_count:
            skipped_count += 1
            continue

        # 6. 准备DST输入：(model_name, confidence)元组列表
        dst_input = []
        binary_masks = []

        for model_idx, inst_id in group:
            model_name = model_names[model_idx]

            # 获取该实例的二值掩码
            mask = (masks_list[model_idx] == inst_id)
            binary_masks.append(mask)

            # 获取置信度
            if confidences_list[model_idx] is not None:
                # 使用该实例区域的平均置信度
                confidence = np.mean(confidences_list[model_idx][mask])
            else:
                # 如果没有置信度图，使用默认值0.8
                confidence = 0.8

            dst_input.append((model_name, float(confidence)))

        # 7. 执行DST融合
        try:
            result: FusionResult = dst_engine.fuse_instances(dst_input)

            # 7.5. 计算形状差异（增强冲突度检测）
            # 如果多个模型的掩码形状差异很大，增加冲突度
            shape_disagreement = 0.0
            if len(binary_masks) >= 2:
                # 计算所有掩码对之间的IoU
                ious = []
                for i in range(len(binary_masks)):
                    for j in range(i + 1, len(binary_masks)):
                        intersection = np.sum(binary_masks[i] & binary_masks[j])
                        union = np.sum(binary_masks[i] | binary_masks[j])
                        if union > 0:
                            iou = intersection / union
                            ious.append(iou)

                if ious:
                    avg_iou = np.mean(ious)
                    # 形状差异 = 1 - 平均IoU
                    # IoU低表示形状差异大
                    shape_disagreement = 1.0 - avg_iou

            # 调整冲突度：结合DST冲突度和形状差异
            # 使用加权平均：DST冲突度权重0.6，形状差异权重0.4
            adjusted_conflict = result.conflict * 0.6 + shape_disagreement * 0.4

            # 8. 处理冲突（使用调整后的冲突度）
            conflict_info = handle_conflict(result, {'low': 0.3, 'medium': conflict_threshold, 'high': 0.9})

            # 9. 根据冲突处理结果决定是否接受
            if conflict_info['action'] in ['accept', 'accept_with_caution']:
                # DST自适应融合策略：根据置信度和冲突度动态调整
                stack = np.stack(binary_masks, axis=0)
                vote_count = np.sum(stack, axis=0)

                # 自适应融合策略：基于置信度×冲突度的二维决策矩阵
                conf = result.confidence
                conflict = adjusted_conflict  # 使用调整后的冲突度（包含形状差异）
                n_models = len(binary_masks)
                strategy_used = ""

                # 策略1: 高置信度区域 (conf > 0.75)
                if conf > 0.75:
                    if conflict < 0.2:
                        # 超高置信度 + 超低冲突 → 超激进（任意1个模型）
                        consensus_mask = vote_count >= 1
                        strategy_used = "ULTRA_AGGRESSIVE"
                    elif conflict < 0.4:
                        # 高置信度 + 低冲突 → 激进（30%阈值）
                        consensus_mask = vote_count >= max(1, int(n_models * 0.3))
                        strategy_used = "AGGRESSIVE"
                    else:
                        # 高置信度 + 中高冲突 → 标准（50%阈值）
                        consensus_mask = vote_count >= (n_models / 2)
                        strategy_used = "STANDARD_HIGH_CONF"

                # 策略2: 中等置信度区域 (0.55 < conf ≤ 0.75)
                elif conf > 0.55:
                    if conflict < 0.3:
                        # 中高置信度 + 低冲突 → 宽松（40%阈值）
                        consensus_mask = vote_count >= max(1, int(n_models * 0.4))
                        strategy_used = "RELAXED"
                    elif conflict < 0.5:
                        # 中等置信度 + 中等冲突 → 标准（50%阈值）
                        consensus_mask = vote_count >= (n_models / 2)
                        strategy_used = "STANDARD_MID_CONF"
                    else:
                        # 中等置信度 + 高冲突 → 严格（60%阈值）
                        consensus_mask = vote_count >= max(1, int(n_models * 0.6))
                        strategy_used = "STRICT"

                # 策略3: 低置信度区域 (conf ≤ 0.55)
                else:
                    if conflict < 0.4:
                        # 低置信度 + 低冲突 → 标准（50%阈值）
                        consensus_mask = vote_count >= (n_models / 2)
                        strategy_used = "STANDARD_LOW_CONF"
                    elif conflict < 0.6:
                        # 低置信度 + 中冲突 → 严格（60%阈值）
                        consensus_mask = vote_count >= max(1, int(n_models * 0.6))
                        strategy_used = "STRICT_LOW_CONF"
                    else:
                        # 低置信度 + 高冲突 → 超严格（70%或交集）
                        if n_models >= 3:
                            consensus_mask = vote_count >= max(2, int(n_models * 0.7))
                            strategy_used = "ULTRA_STRICT"
                        else:
                            consensus_mask = vote_count == n_models
                            strategy_used = "INTERSECTION"

                logger.debug(f"组{group_idx}: conf={result.confidence:.3f}, conflict={result.conflict:.3f}, "
                           f"strategy={strategy_used}, models={len(binary_masks)}")

                # 统计策略使用
                strategy_counts[strategy_used] += 1

                if np.sum(consensus_mask) > 0:
                    fused_mask[consensus_mask] = instance_id

                    # 记录融合结果
                    fusion_results.append({
                        'instance_id': instance_id,
                        'group_idx': group_idx,
                        'decision': result.decision,
                        'confidence': result.confidence,
                        'conflict': adjusted_conflict,  # 使用调整后的冲突度
                        'dst_conflict': result.conflict,  # 保留原始DST冲突度
                        'shape_disagreement': shape_disagreement,  # 添加形状差异
                        'uncertainty': result.uncertainty,
                        'belief_cell': result.belief_cell,
                        'plausibility_cell': result.plausibility_cell,
                        'status': conflict_info['status'],
                        'strategy': strategy_used  # 添加策略信息
                    })

                    instance_id += 1
                    fused_count += 1

                    # 标记高冲突实例（使用调整后的冲突度）
                    if adjusted_conflict >= conflict_threshold:
                        high_conflict_count += 1
                        high_conflict_instances.append({
                            'instance_id': instance_id - 1,
                            'conflict': adjusted_conflict,
                            'dst_conflict': result.conflict,
                            'shape_disagreement': shape_disagreement,
                            'uncertainty': result.uncertainty
                        })
            else:
                # 拒绝或需要人工审查
                logger.warning(f"组{group_idx}被拒绝：{conflict_info.get('warning', conflict_info.get('error'))}")
                skipped_count += 1

        except Exception as e:
            logger.error(f"组{group_idx}的DST融合失败: {e}")
            skipped_count += 1

    # 10. 生成统计信息
    # 计算置信度和冲突度的分布
    confidences = [r['confidence'] for r in fusion_results]
    conflicts = [r['conflict'] for r in fusion_results]

    dst_stats = {
        'total_groups': total_groups,
        'fused_count': fused_count,
        'skipped_count': skipped_count,
        'high_conflict_count': high_conflict_count,
        'fusion_results': fusion_results,
        'high_conflict_instances': high_conflict_instances,
        'average_conflict': np.mean(conflicts) if conflicts else 0.0,
        'average_uncertainty': np.mean([r['uncertainty'] for r in fusion_results]) if fusion_results else 0.0,
        'strategy_counts': strategy_counts,  # 添加策略统计
        # 添加置信度和冲突度分布统计
        'confidence_distribution': {
            'min': np.min(confidences) if confidences else 0.0,
            'max': np.max(confidences) if confidences else 0.0,
            'mean': np.mean(confidences) if confidences else 0.0,
            'std': np.std(confidences) if confidences else 0.0
        },
        'conflict_distribution': {
            'min': np.min(conflicts) if conflicts else 0.0,
            'max': np.max(conflicts) if conflicts else 0.0,
            'mean': np.mean(conflicts) if conflicts else 0.0,
            'std': np.std(conflicts) if conflicts else 0.0
        }
    }

    logger.info(f"DST融合完成：总组数={total_groups}，融合={fused_count}，跳过={skipped_count}，"
               f"高冲突={high_conflict_count}，平均冲突={dst_stats['average_conflict']:.3f}")

    # 输出置信度和冲突度分布
    logger.info(f"置信度分布：min={dst_stats['confidence_distribution']['min']:.3f}, "
               f"max={dst_stats['confidence_distribution']['max']:.3f}, "
               f"mean={dst_stats['confidence_distribution']['mean']:.3f}, "
               f"std={dst_stats['confidence_distribution']['std']:.3f}")
    logger.info(f"冲突度分布：min={dst_stats['conflict_distribution']['min']:.3f}, "
               f"max={dst_stats['conflict_distribution']['max']:.3f}, "
               f"mean={dst_stats['conflict_distribution']['mean']:.3f}, "
               f"std={dst_stats['conflict_distribution']['std']:.3f}")

    # 输出策略使用统计
    logger.info(f"策略使用统计：")
    for strategy, count in strategy_counts.items():
        if count > 0:
            percentage = (count / fused_count * 100) if fused_count > 0 else 0
            logger.info(f"  {strategy}: {count} ({percentage:.1f}%)")

    # 11. 应用分水岭边界细化（如果启用且有融合结果）
    if enable_watershed and fused_count > 0 and image is not None:
        logger.info(f"开始分水岭边界细化，融合实例数={fused_count}，高冲突实例数={high_conflict_count}")

        # 创建实例级冲突图
        conflict_map_instance = np.zeros_like(fused_mask, dtype=np.float32)
        for result in fusion_results:
            instance_mask = fused_mask == result['instance_id']
            conflict_map_instance[instance_mask] = result['conflict']

        # 应用分水岭细化
        from .geometric_refinement import watershed_refinement
        refined_mask, refinement_stats = watershed_refinement(
            image=image,
            fused_mask=fused_mask,
            conflict_map=conflict_map_instance,
            conflict_threshold=conflict_threshold,
            gradient_sigma=1.0
        )

        # 更新统计信息
        dst_stats['watershed_refinement'] = refinement_stats

        if refinement_stats['refined']:
            logger.info(f"分水岭细化成功：修改了 {refinement_stats['refined_pixels']} 个像素 "
                       f"({refinement_stats['refined_percentage']:.2f}%)")
            fused_mask = refined_mask
        else:
            logger.warning(f"分水岭细化失败：{refinement_stats.get('reason', 'unknown')}")
    elif enable_watershed and high_conflict_count > 0 and image is None:
        logger.warning("分水岭细化已启用但未提供原始图像，跳过细化步骤")
        dst_stats['watershed_refinement'] = {'refined': False, 'reason': 'no_image'}
    else:
        dst_stats['watershed_refinement'] = {'refined': False, 'reason': 'disabled_or_no_conflict'}

    return fused_mask, dst_stats
