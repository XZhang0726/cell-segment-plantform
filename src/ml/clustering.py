"""
聚类分析模块

提供多种无监督聚类算法，用于细胞形态学特征的分类分析
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from loguru import logger


def preprocess_features(features_df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """
    预处理特征数据：排除非特征列并进行标准化

    Args:
        features_df: 特征DataFrame
        exclude_cols: 需要排除的列名列表（如果为None，使用默认排除列表）

    Returns:
        features_scaled: 标准化后的特征数组
        feature_cols: 使用的特征列名列表
        scaler: 标准化器对象（用于逆变换）
    """
    if exclude_cols is None:
        # 默认排除的列：ID、位置、边界框
        exclude_cols = [
            'sequential_id', 'cell_id',
            'centroid_x', 'centroid_y',
            'bbox_min_row', 'bbox_min_col',
            'bbox_max_row', 'bbox_max_col'
        ]

    # 获取特征列
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]

    # 提取特征数据
    features = features_df[feature_cols].values

    # 检查是否有NaN或Inf
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.warning("Features contain NaN or Inf values, replacing with column means")
        features = pd.DataFrame(features, columns=feature_cols).fillna(method='ffill').fillna(0).values

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    logger.info(f"Preprocessed {len(feature_cols)} features for {len(features_df)} cells")

    return features_scaled, feature_cols, scaler


def perform_kmeans(features_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    执行K-means聚类

    Args:
        features_df: 特征DataFrame
        n_clusters: 聚类数量
        random_state: 随机种子

    Returns:
        labels: 聚类标签数组
        info: 包含聚类信息的字典（模型、惯性等）
    """
    # 预处理特征
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    # 计算聚类质量指标
    silhouette = silhouette_score(features_scaled, labels) if n_clusters > 1 else 0
    davies_bouldin = davies_bouldin_score(features_scaled, labels) if n_clusters > 1 else 0
    calinski_harabasz = calinski_harabasz_score(features_scaled, labels) if n_clusters > 1 else 0

    info = {
        'model': kmeans,
        'inertia': kmeans.inertia_,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'n_clusters': n_clusters,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"K-means clustering completed: {n_clusters} clusters, silhouette={silhouette:.3f}")

    return labels, info


def perform_dbscan(features_df: pd.DataFrame, eps: Optional[float] = None, min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
    """
    执行DBSCAN聚类（基于密度的聚类）

    Args:
        features_df: 特征DataFrame
        eps: 邻域半径（如果为None，自动估计）
        min_samples: 核心点的最小邻居数

    Returns:
        labels: 聚类标签数组（-1表示噪声点）
        info: 包含聚类信息的字典
    """
    # 预处理特征
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # 如果eps未指定，使用k-distance图自动估计
    if eps is None:
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(features_scaled)
        distances, _ = neighbors.kneighbors(features_scaled)
        distances = np.sort(distances[:, -1])
        # 使用90th百分位数作为eps
        eps = np.percentile(distances, 90)
        logger.info(f"Auto-estimated eps={eps:.3f}")

    # 执行DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features_scaled)

    # 计算聚类数量（排除噪声点）
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # 计算聚类质量指标（排除噪声点）
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 0:
            silhouette = silhouette_score(features_scaled[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(features_scaled[mask], labels[mask])
            calinski_harabasz = calinski_harabasz_score(features_scaled[mask], labels[mask])
        else:
            silhouette = davies_bouldin = calinski_harabasz = 0
    else:
        silhouette = davies_bouldin = calinski_harabasz = 0

    info = {
        'model': dbscan,
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"DBSCAN clustering completed: {n_clusters} clusters, {n_noise} noise points, silhouette={silhouette:.3f}")

    return labels, info


def perform_hierarchical(features_df: pd.DataFrame, n_clusters: int = 3, linkage: str = 'ward') -> Tuple[np.ndarray, Dict]:
    """
    执行层次聚类

    Args:
        features_df: 特征DataFrame
        n_clusters: 聚类数量
        linkage: 链接方法 ('ward', 'complete', 'average', 'single')

    Returns:
        labels: 聚类标签数组
        info: 包含聚类信息的字典
    """
    # 预处理特征
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # 执行层次聚类
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hierarchical.fit_predict(features_scaled)

    # 计算聚类质量指标
    silhouette = silhouette_score(features_scaled, labels) if n_clusters > 1 else 0
    davies_bouldin = davies_bouldin_score(features_scaled, labels) if n_clusters > 1 else 0
    calinski_harabasz = calinski_harabasz_score(features_scaled, labels) if n_clusters > 1 else 0

    info = {
        'model': hierarchical,
        'n_clusters': n_clusters,
        'linkage': linkage,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"Hierarchical clustering completed: {n_clusters} clusters, linkage={linkage}, silhouette={silhouette:.3f}")

    return labels, info


def perform_gmm(features_df: pd.DataFrame, n_components: int = 3, random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    执行高斯混合模型聚类

    Args:
        features_df: 特征DataFrame
        n_components: 高斯分量数量
        random_state: 随机种子

    Returns:
        labels: 聚类标签数组
        info: 包含聚类信息的字典
    """
    # 预处理特征
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # 执行GMM聚类
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, covariance_type='full')
    labels = gmm.fit_predict(features_scaled)

    # 计算聚类质量指标
    silhouette = silhouette_score(features_scaled, labels) if n_components > 1 else 0
    davies_bouldin = davies_bouldin_score(features_scaled, labels) if n_components > 1 else 0
    calinski_harabasz = calinski_harabasz_score(features_scaled, labels) if n_components > 1 else 0

    info = {
        'model': gmm,
        'n_components': n_components,
        'bic': gmm.bic(features_scaled),
        'aic': gmm.aic(features_scaled),
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"GMM clustering completed: {n_components} components, BIC={gmm.bic(features_scaled):.2f}, silhouette={silhouette:.3f}")

    return labels, info


def evaluate_clustering(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    评估聚类质量

    Args:
        features: 特征数组
        labels: 聚类标签数组

    Returns:
        包含评估指标的字典
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2:
        return {
            'silhouette_score': 0.0,
            'davies_bouldin_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'n_clusters': n_clusters
        }

    # 排除噪声点（DBSCAN可能产生-1标签）
    mask = labels != -1
    if mask.sum() == 0:
        return {
            'silhouette_score': 0.0,
            'davies_bouldin_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'n_clusters': 0
        }

    features_clean = features[mask]
    labels_clean = labels[mask]

    # 计算评估指标
    silhouette = silhouette_score(features_clean, labels_clean)
    davies_bouldin = davies_bouldin_score(features_clean, labels_clean)
    calinski_harabasz = calinski_harabasz_score(features_clean, labels_clean)

    return {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'n_clusters': n_clusters
    }


def find_optimal_clusters(features_df: pd.DataFrame, max_k: int = 10, method: str = 'kmeans') -> Dict:
    """
    寻找最佳聚类数量

    Args:
        features_df: 特征DataFrame
        max_k: 最大聚类数量
        method: 聚类方法 ('kmeans', 'gmm')

    Returns:
        包含不同k值评估结果的字典
    """
    # 预处理特征
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    results = {
        'k_values': [],
        'inertias': [],
        'silhouette_scores': [],
        'davies_bouldin_scores': [],
        'calinski_harabasz_scores': []
    }

    if method == 'gmm':
        results['bic_scores'] = []
        results['aic_scores'] = []

    logger.info(f"Finding optimal number of clusters (k=2 to {max_k})...")

    for k in range(2, max_k + 1):
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            inertia = kmeans.inertia_
            results['inertias'].append(inertia)
        elif method == 'gmm':
            gmm = GaussianMixture(n_components=k, random_state=42)
            labels = gmm.fit_predict(features_scaled)
            results['bic_scores'].append(gmm.bic(features_scaled))
            results['aic_scores'].append(gmm.aic(features_scaled))

        # 计算评估指标
        silhouette = silhouette_score(features_scaled, labels)
        davies_bouldin = davies_bouldin_score(features_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(features_scaled, labels)

        results['k_values'].append(k)
        results['silhouette_scores'].append(silhouette)
        results['davies_bouldin_scores'].append(davies_bouldin)
        results['calinski_harabasz_scores'].append(calinski_harabasz)

    # 找到最佳k值（基于轮廓系数）
    best_k_idx = np.argmax(results['silhouette_scores'])
    best_k = results['k_values'][best_k_idx]

    results['best_k'] = best_k
    results['best_silhouette'] = results['silhouette_scores'][best_k_idx]

    logger.info(f"Optimal k={best_k} with silhouette score={results['best_silhouette']:.3f}")

    return results
