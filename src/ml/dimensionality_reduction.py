"""
降维可视化模块

提供多种降维方法，将高维特征降至2D/3D用于可视化
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from loguru import logger

# UMAP导入（可选依赖）
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")


def apply_pca(features_df: pd.DataFrame, n_components: int = 2, exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, PCA, List[str]]:
    """
    应用PCA降维

    Args:
        features_df: 特征DataFrame
        n_components: 降维后的维度（2或3）
        exclude_cols: 需要排除的列名列表

    Returns:
        components: 降维后的数据
        pca: PCA模型对象
        feature_cols: 使用的特征列名列表
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

    # 检查并处理NaN/Inf
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.warning("Features contain NaN or Inf, replacing with column means")
        features = pd.DataFrame(features, columns=feature_cols).fillna(method='ffill').fillna(0).values

    # 应用PCA
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(features)

    # 计算解释方差比例
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()

    logger.info(f"PCA completed: {n_components} components explain {total_variance*100:.2f}% of variance")

    return components, pca, feature_cols


def apply_tsne(features_df: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0,
               n_iter: int = 1000, exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    应用t-SNE降维

    Args:
        features_df: 特征DataFrame
        n_components: 降维后的维度（2或3）
        perplexity: 困惑度参数（5-50之间，默认30）
        n_iter: 迭代次数
        exclude_cols: 需要排除的列名列表

    Returns:
        components: 降维后的数据
        feature_cols: 使用的特征列名列表
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

    # 检查并处理NaN/Inf
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.warning("Features contain NaN or Inf, replacing with column means")
        features = pd.DataFrame(features, columns=feature_cols).fillna(method='ffill').fillna(0).values

    # 调整perplexity以适应样本数量
    n_samples = len(features)
    if perplexity >= n_samples:
        perplexity = max(5, n_samples // 3)
        logger.warning(f"Perplexity too large for {n_samples} samples, adjusted to {perplexity}")

    # 应用t-SNE
    logger.info(f"Running t-SNE (this may take a while for large datasets)...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=n_iter,
                random_state=42, verbose=0)
    components = tsne.fit_transform(features)

    logger.info(f"t-SNE completed: {n_components} components, perplexity={perplexity}")

    return components, feature_cols


def apply_umap(features_df: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15,
               min_dist: float = 0.1, exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    应用UMAP降维

    Args:
        features_df: 特征DataFrame
        n_components: 降维后的维度（2或3）
        n_neighbors: 邻居数量（5-50之间，默认15）
        min_dist: 最小距离（0.0-0.99之间，默认0.1）
        exclude_cols: 需要排除的列名列表

    Returns:
        components: 降维后的数据
        feature_cols: 使用的特征列名列表
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is not installed. Install with: pip install umap-learn")

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

    # 检查并处理NaN/Inf
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.warning("Features contain NaN or Inf, replacing with column means")
        features = pd.DataFrame(features, columns=feature_cols).fillna(method='ffill').fillna(0).values

    # 调整n_neighbors以适应样本数量
    n_samples = len(features)
    if n_neighbors >= n_samples:
        n_neighbors = max(2, n_samples // 2)
        logger.warning(f"n_neighbors too large for {n_samples} samples, adjusted to {n_neighbors}")

    # 应用UMAP
    logger.info(f"Running UMAP...")
    umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                      random_state=42, verbose=False)
    components = umap_model.fit_transform(features)

    logger.info(f"UMAP completed: {n_components} components, n_neighbors={n_neighbors}, min_dist={min_dist}")

    return components, feature_cols
