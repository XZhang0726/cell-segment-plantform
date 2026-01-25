"""
特征分析模块

提供特征重要性分析、相关性计算和特征选择功能
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
from loguru import logger


def analyze_feature_importance(features_df: pd.DataFrame, cluster_labels: np.ndarray,
                               exclude_cols: Optional[List[str]] = None, top_n: int = 20) -> pd.DataFrame:
    """
    分析特征对聚类的重要性

    使用随机森林分类器评估每个特征对聚类结果的贡献

    Args:
        features_df: 特征DataFrame
        cluster_labels: 聚类标签数组
        exclude_cols: 需要排除的列名列表
        top_n: 返回前N个最重要的特征

    Returns:
        包含特征重要性的DataFrame
    """
    if exclude_cols is None:
        exclude_cols = [
            'sequential_id', 'cell_id',
            'centroid_x', 'centroid_y',
            'bbox_min_row', 'bbox_min_col',
            'bbox_max_row', 'bbox_max_col'
        ]

    # 获取特征列
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    features = features_df[feature_cols].values

    # 检查聚类数量
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    if n_clusters < 2:
        logger.warning("Less than 2 clusters, cannot analyze feature importance")
        return pd.DataFrame()

    # 排除噪声点（DBSCAN可能产生-1标签）
    mask = cluster_labels != -1
    features_clean = features[mask]
    labels_clean = cluster_labels[mask]

    if len(features_clean) == 0:
        logger.warning("No valid samples after removing noise points")
        return pd.DataFrame()

    # 训练随机森林分类器
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(features_clean, labels_clean)

    # 获取特征重要性
    importances = rf.feature_importances_

    # 创建结果DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    })

    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # 只返回前N个
    importance_df = importance_df.head(top_n)

    logger.info(f"Feature importance analysis completed: top {len(importance_df)} features identified")

    return importance_df


def compute_feature_correlation(features_df: pd.DataFrame, exclude_cols: Optional[List[str]] = None,
                                method: str = 'spearman') -> pd.DataFrame:
    """
    计算特征之间的相关性

    Args:
        features_df: 特征DataFrame
        exclude_cols: 需要排除的列名列表
        method: 相关性计算方法 ('pearson', 'spearman')

    Returns:
        相关性矩阵DataFrame
    """
    if exclude_cols is None:
        exclude_cols = [
            'sequential_id', 'cell_id',
            'centroid_x', 'centroid_y',
            'bbox_min_row', 'bbox_min_col',
            'bbox_max_row', 'bbox_max_col'
        ]

    # 获取特征列
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    features = features_df[feature_cols]

    # 计算相关性矩阵
    if method == 'pearson':
        corr_matrix = features.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = features.corr(method='spearman')
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    logger.info(f"Feature correlation matrix computed using {method} method")

    return corr_matrix


def select_top_features(features_df: pd.DataFrame, cluster_labels: np.ndarray,
                       n_features: int = 20, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    选择最重要的特征

    基于特征重要性和相关性，选择最具代表性的特征子集

    Args:
        features_df: 特征DataFrame
        cluster_labels: 聚类标签数组
        n_features: 要选择的特征数量
        exclude_cols: 需要排除的列名列表

    Returns:
        选中的特征列名列表
    """
    # 分析特征重要性
    importance_df = analyze_feature_importance(features_df, cluster_labels, exclude_cols, top_n=n_features*2)

    if importance_df.empty:
        logger.warning("Cannot select features: importance analysis failed")
        return []

    # 获取高重要性特征
    top_features = importance_df['feature'].tolist()

    # 计算这些特征之间的相关性
    if exclude_cols is None:
        exclude_cols = []

    # 只保留top_features中的特征
    features_subset = features_df[[f for f in top_features if f in features_df.columns]]
    corr_matrix = features_subset.corr(method='spearman').abs()

    # 移除高度相关的特征（保留重要性更高的）
    selected_features = []
    for feature in top_features:
        if feature not in features_subset.columns:
            continue

        # 检查是否与已选特征高度相关
        is_redundant = False
        for selected in selected_features:
            if selected in corr_matrix.columns and feature in corr_matrix.index:
                if corr_matrix.loc[feature, selected] > 0.9:  # 相关性阈值
                    is_redundant = True
                    break

        if not is_redundant:
            selected_features.append(feature)

        # 达到目标数量
        if len(selected_features) >= n_features:
            break

    logger.info(f"Selected {len(selected_features)} features from {len(top_features)} candidates")

    return selected_features
