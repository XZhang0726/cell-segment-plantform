"""
机器学习模块

提供无监督学习功能，包括聚类分析、降维可视化和特征分析
"""
from .clustering import (
    perform_kmeans,
    perform_dbscan,
    perform_hierarchical,
    perform_gmm,
    evaluate_clustering,
    find_optimal_clusters,
    preprocess_features
)

from .dimensionality_reduction import (
    apply_pca,
    apply_tsne,
    apply_umap
)

from .feature_analysis import (
    analyze_feature_importance,
    compute_feature_correlation,
    select_top_features
)

__all__ = [
    # 聚类分析
    'perform_kmeans',
    'perform_dbscan',
    'perform_hierarchical',
    'perform_gmm',
    'evaluate_clustering',
    'find_optimal_clusters',
    'preprocess_features',

    # 降维可视化
    'apply_pca',
    'apply_tsne',
    'apply_umap',

    # 特征分析
    'analyze_feature_importance',
    'compute_feature_correlation',
    'select_top_features'
]
