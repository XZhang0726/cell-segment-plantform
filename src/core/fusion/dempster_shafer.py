"""
Dempster-Shafer Theory (DST) Fusion for Cell Instance Segmentation

This module implements evidence-based fusion using Dempster-Shafer Theory,
which provides a mathematical framework for combining evidence from multiple
segmentation models while explicitly modeling uncertainty and conflict.

Key Concepts:
- Mass Function: Represents evidence distribution over hypotheses
- Dempster's Combination Rule: Combines evidence from multiple sources
- Conflict Coefficient: Quantifies disagreement between models
- Belief/Plausibility: Lower and upper bounds of probability

Author: Claude Sonnet 4.5
Date: 2026-01-27
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class FusionResult:
    """融合结果数据类

    Attributes:
        mass_function: 最终质量函数字典
        decision: 决策结果 ('Cell', 'Background', 'Cell|Background')
        confidence: 置信度 [0,1]
        conflict: 冲突系数 [0,1]
        uncertainty: 不确定性 [0,1]
        belief_cell: Cell的信念值 [0,1]
        plausibility_cell: Cell的似然值 [0,1]
    """
    mass_function: Dict[str, float]
    decision: str
    confidence: float
    conflict: float
    uncertainty: float
    belief_cell: float
    plausibility_cell: float


class DempsterShaferFusion:
    """
    Dempster-Shafer理论融合引擎

    用于融合多个细胞分割模型的预测结果，提供比简单投票更严格的
    数学框架来处理不确定性和模型冲突。

    Example:
        >>> fusion_engine = DempsterShaferFusion({
        ...     'cellpose': 0.9,
        ...     'cellvit': 0.85,
        ...     'cellsam': 0.8
        ... })
        >>> matched_group = [
        ...     ('cellpose', 0.8),
        ...     ('cellvit', 0.6),
        ...     ('cellsam', 0.9)
        ... ]
        >>> result = fusion_engine.fuse_instances(matched_group)
        >>> print(f"Decision: {result.decision}, Confidence: {result.confidence:.3f}")
    """

    def __init__(self, model_reliabilities: Dict[str, float]):
        """
        初始化融合引擎

        Args:
            model_reliabilities: 每个模型的可靠性参数 [0,1]
                例如: {'cellpose': 0.9, 'cellvit': 0.85, 'cellsam': 0.8}
                可靠性越高，模型的证据权重越大
        """
        self.reliabilities = model_reliabilities
        self.hypotheses = ['Cell', 'Background', 'Cell|Background']
        logger.info(f"Initialized DST Fusion Engine with reliabilities: {model_reliabilities}")

    def compute_mass_from_confidence(self,
                                    model_name: str,
                                    confidence: float) -> Dict[str, float]:
        """
        从置信度计算质量函数

        将模型输出的置信度分数转换为DST质量函数。

        公式:
            m(Cell) = confidence × reliability
            m(Background) = (1 - confidence) × reliability
            m(Cell|Background) = 1 - reliability

        Args:
            model_name: 模型名称
            confidence: 置信度 [0,1]

        Returns:
            质量函数字典，键为假设，值为质量值
        """
        reliability = self.reliabilities.get(model_name, 0.8)

        mass_func = {
            'Cell': confidence * reliability,
            'Background': (1 - confidence) * reliability,
            'Cell|Background': 1 - reliability
        }

        # 验证质量函数和为1
        total = sum(mass_func.values())
        assert abs(total - 1.0) < 1e-6, f"Mass function sum is {total}, should be 1.0"

        return mass_func

    def compute_intersection(self, A: str, B: str) -> str:
        """
        计算两个假设的交集

        交集规则:
        - Cell ∩ Cell = Cell
        - Cell ∩ Background = ∅ (冲突)
        - Cell ∩ Uncertain = Cell
        - Background ∩ Background = Background
        - Uncertain ∩ Uncertain = Uncertain

        Args:
            A, B: 假设字符串，例如 'Cell', 'Background', 'Cell|Background'

        Returns:
            交集字符串，'∅' 表示空集（冲突）
        """
        # 解析假设为集合
        set_A = set(A.split('|'))
        set_B = set(B.split('|'))

        # 计算交集
        intersection = set_A & set_B

        if len(intersection) == 0:
            return '∅'  # 空集，表示冲突
        else:
            return '|'.join(sorted(intersection))

    def dempster_combine(self,
                        m1: Dict[str, float],
                        m2: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """
        Dempster组合规则

        组合两个质量函数，计算联合证据和冲突系数。

        数学公式:
            m₁₂(C) = [Σ m₁(A) × m₂(B)] / (1 - K)
                     A∩B=C

            K = Σ m₁(A) × m₂(B)  (冲突系数)
                A∩B=∅

        Args:
            m1, m2: 两个质量函数

        Returns:
            (组合后的质量函数, 冲突系数)

        Raises:
            ValueError: 如果冲突系数 >= 1.0 (完全冲突)
        """
        combined = {}
        conflict = 0.0

        # 计算所有可能的交集
        for (A, m1_A) in m1.items():
            for (B, m2_B) in m2.items():
                intersection = self.compute_intersection(A, B)

                if intersection == '∅':
                    # 冲突：两个假设不相容
                    conflict += m1_A * m2_B
                else:
                    # 累加到对应的交集
                    if intersection not in combined:
                        combined[intersection] = 0.0
                    combined[intersection] += m1_A * m2_B

        # 检查完全冲突
        if conflict >= 1.0:
            logger.error(f"Complete conflict detected! K={conflict}")
            raise ValueError(f"完全冲突！K={conflict}，无法融合")

        # 归一化
        normalization = 1.0 - conflict
        for key in combined:
            combined[key] /= normalization

        logger.debug(f"Dempster combination: conflict={conflict:.3f}, combined={combined}")

        return combined, conflict

    def combine_multiple(self,
                        mass_functions: List[Dict[str, float]]) -> Tuple[Dict[str, float], float]:
        """
        组合多个质量函数

        迭代应用Dempster组合规则，将多个模型的证据融合为单一的质量函数。

        Args:
            mass_functions: 质量函数列表

        Returns:
            (最终组合的质量函数, 累积冲突系数)

        Raises:
            ValueError: 如果质量函数列表为空
        """
        if len(mass_functions) == 0:
            raise ValueError("至少需要一个质量函数")

        if len(mass_functions) == 1:
            return mass_functions[0], 0.0

        # 迭代组合
        combined = mass_functions[0]
        total_conflict = 0.0

        for i in range(1, len(mass_functions)):
            combined, conflict = self.dempster_combine(combined, mass_functions[i])
            total_conflict += conflict

        logger.info(f"Combined {len(mass_functions)} mass functions, total conflict: {total_conflict:.3f}")

        return combined, total_conflict

    def compute_belief(self,
                      mass_function: Dict[str, float],
                      hypothesis: str) -> float:
        """
        计算信念函数 Bel(A)

        信念函数表示对假设A的最小支持度（下界）。

        公式:
            Bel(A) = Σ m(B), 对所有 B ⊆ A

        Args:
            mass_function: 质量函数
            hypothesis: 目标假设

        Returns:
            信念值 [0,1]
        """
        belief = 0.0
        target_set = set(hypothesis.split('|'))

        for B, mass in mass_function.items():
            B_set = set(B.split('|'))
            if B_set.issubset(target_set):
                belief += mass

        return belief

    def compute_plausibility(self,
                            mass_function: Dict[str, float],
                            hypothesis: str) -> float:
        """
        计算似然函数 Pl(A)

        似然函数表示对假设A的最大支持度（上界）。

        公式:
            Pl(A) = Σ m(B), 对所有 B ∩ A ≠ ∅

        Args:
            mass_function: 质量函数
            hypothesis: 目标假设

        Returns:
            似然值 [0,1]
        """
        plausibility = 0.0
        target_set = set(hypothesis.split('|'))

        for B, mass in mass_function.items():
            B_set = set(B.split('|'))
            if len(B_set & target_set) > 0:  # 有交集
                plausibility += mass

        return plausibility

    def decide(self, mass_function: Dict[str, float]) -> Tuple[str, float]:
        """
        决策：选择最大质量的单元素假设

        从质量函数中选择具有最大质量值的单元素假设作为最终决策。

        Args:
            mass_function: 质量函数

        Returns:
            (决策结果, 置信度)
        """
        max_mass = 0.0
        decision = 'Cell|Background'  # 默认不确定

        for hypothesis, mass in mass_function.items():
            if '|' not in hypothesis:  # 单元素假设
                if mass > max_mass:
                    max_mass = mass
                    decision = hypothesis

        return decision, max_mass

    def fuse_instances(self,
                      matched_group: List[Tuple[str, float]]) -> FusionResult:
        """
        融合一组匹配的实例

        这是主要的融合接口，接受一组匹配的实例（来自不同模型），
        返回融合结果和相关的不确定性度量。

        Args:
            matched_group: [(model_name, confidence), ...]
                例如: [('cellpose', 0.8), ('cellvit', 0.6), ('cellsam', 0.9)]

        Returns:
            FusionResult对象，包含决策、置信度、冲突度等信息
        """
        logger.debug(f"Fusing {len(matched_group)} instances: {matched_group}")

        # 步骤1: 计算每个模型的质量函数
        mass_functions = []
        for model_name, confidence in matched_group:
            mass = self.compute_mass_from_confidence(model_name, confidence)
            mass_functions.append(mass)
            logger.debug(f"  {model_name}: confidence={confidence:.3f}, mass={mass}")

        # 步骤2: 组合所有质量函数
        combined_mass, total_conflict = self.combine_multiple(mass_functions)

        # 步骤3: 决策
        decision, confidence = self.decide(combined_mass)

        # 步骤4: 计算信念和似然
        belief_cell = self.compute_belief(combined_mass, 'Cell')
        plausibility_cell = self.compute_plausibility(combined_mass, 'Cell')

        # 步骤5: 计算不确定性
        uncertainty = combined_mass.get('Cell|Background', 0.0)

        result = FusionResult(
            mass_function=combined_mass,
            decision=decision,
            confidence=confidence,
            conflict=total_conflict,
            uncertainty=uncertainty,
            belief_cell=belief_cell,
            plausibility_cell=plausibility_cell
        )

        logger.info(f"DST Fusion Result: decision={decision}, confidence={confidence:.3f}, "
                   f"conflict={total_conflict:.3f}, uncertainty={uncertainty:.3f}")

        return result


def handle_conflict(result: FusionResult,
                   conflict_thresholds: Optional[Dict[str, float]] = None) -> Dict:
    """
    根据冲突程度采取不同策略

    冲突分级:
    - 低冲突 (K < 0.3): 正常融合
    - 中等冲突 (0.3 ≤ K < 0.6): 降低置信度
    - 高冲突 (0.6 ≤ K < 0.9): 标记需要人工审查
    - 完全冲突 (K ≥ 0.9): 拒绝融合

    Args:
        result: 融合结果
        conflict_thresholds: 冲突阈值字典

    Returns:
        处理后的结果字典，包含状态、决策、置信度和建议操作
    """
    if conflict_thresholds is None:
        conflict_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9
        }

    K = result.conflict

    if K < conflict_thresholds['low']:
        # 低冲突：正常
        return {
            'status': 'NORMAL',
            'decision': result.decision,
            'confidence': result.confidence,
            'action': 'accept',
            'message': f'低冲突 (K={K:.3f})，融合结果可靠'
        }

    elif K < conflict_thresholds['medium']:
        # 中等冲突：降低置信度
        adjusted_confidence = result.confidence * (1 - K * 0.5)
        return {
            'status': 'MEDIUM_CONFLICT',
            'decision': result.decision,
            'confidence': adjusted_confidence,
            'action': 'accept_with_caution',
            'warning': f'中等冲突 (K={K:.3f})，置信度已调整'
        }

    elif K < conflict_thresholds['high']:
        # 高冲突：标记审查
        return {
            'status': 'HIGH_CONFLICT',
            'decision': 'Uncertain',
            'confidence': 0.0,
            'action': 'manual_review',
            'warning': f'高冲突 (K={K:.3f})，需要人工审查'
        }

    else:
        # 完全冲突：拒绝
        return {
            'status': 'COMPLETE_CONFLICT',
            'decision': 'Error',
            'confidence': 0.0,
            'action': 'reject',
            'error': f'完全冲突 (K={K:.3f})，无法融合'
        }


def generate_conflict_map(masks_list: List[np.ndarray],
                         confidences_list: List[np.ndarray],
                         model_names: List[str],
                         fusion_engine: DempsterShaferFusion) -> np.ndarray:
    """
    生成像素级冲突图

    对图像中的每个像素，计算多个模型预测的冲突系数，
    生成一张冲突热力图，用于可视化模型分歧区域。

    Args:
        masks_list: 模型分割结果列表 [(H,W), ...]
        confidences_list: 置信度图列表 [(H,W), ...]
        model_names: 模型名称列表
        fusion_engine: DST融合引擎

    Returns:
        冲突图 (H, W)，值范围[0, 1]，值越大表示冲突越严重
    """
    H, W = masks_list[0].shape
    conflict_map = np.zeros((H, W), dtype=np.float32)

    logger.info(f"Generating conflict map for {H}x{W} image with {len(masks_list)} models")

    for i in range(H):
        for j in range(W):
            # 提取该像素的所有模型预测
            pixel_predictions = []
            for k, (mask, conf, name) in enumerate(zip(masks_list,
                                                       confidences_list,
                                                       model_names)):
                if mask[i, j] > 0:  # 该模型认为是前景
                    confidence = conf[i, j] if conf is not None else 0.5
                    pixel_predictions.append((name, confidence))

            # 如果至少有2个模型预测
            if len(pixel_predictions) >= 2:
                try:
                    result = fusion_engine.fuse_instances(pixel_predictions)
                    conflict_map[i, j] = result.conflict
                except Exception as e:
                    logger.warning(f"Fusion failed at pixel ({i},{j}): {e}")
                    conflict_map[i, j] = 1.0  # 融合失败，标记为最高冲突

    logger.info(f"Conflict map generated: mean={np.mean(conflict_map):.3f}, "
               f"max={np.max(conflict_map):.3f}")

    return conflict_map
