"""
多模型分割融合模块

提供实例匹配、融合引擎和不确定性计算功能
包括简单融合策略和高级融合方法（Dempster-Shafer理论）
"""

# 使用优化版本的实例匹配（57倍加速）
from .instance_matcher_optimized import match_instances_optimized as match_instances
from .fusion_engine import fuse_instances, fuse_instances_dst
from .uncertainty import compute_disagreement_map, compute_model_consistency

# 保留原始版本的compute_iou用于兼容性
from .instance_matcher import compute_iou

# 高级融合方法：Dempster-Shafer理论
from .dempster_shafer import (
    DempsterShaferFusion,
    FusionResult,
    handle_conflict,
    generate_conflict_map
)

# 置信度工具
from .confidence_utils import generate_confidence_maps

# 几何辅助细化
from .geometric_refinement import watershed_refinement, compute_gradient_map

__all__ = [
    # 基础功能
    'compute_iou',
    'match_instances',

    # 简单融合策略
    'fuse_instances',

    # 高级融合方法（DST）
    'fuse_instances_dst',
    'DempsterShaferFusion',
    'FusionResult',
    'handle_conflict',
    'generate_conflict_map',

    # 置信度工具
    'generate_confidence_maps',

    # 几何辅助细化
    'watershed_refinement',
    'compute_gradient_map',

    # 不确定性计算
    'compute_disagreement_map',
    'compute_model_consistency'
]
