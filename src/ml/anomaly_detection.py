"""
异常检测模块

提供多种异常检测算法用于识别形态学参数异常的细胞样本
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from loguru import logger


def preprocess_features(features_df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """
    预处理特征数据

    Args:
        features_df: 特征DataFrame
        exclude_cols: 需要排除的列名列表

    Returns:
        features_scaled: 标准化后的特征数组
        feature_cols: 特征列名列表
        scaler: 标准化器
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
        features_df_clean = pd.DataFrame(features, columns=feature_cols)
        features = features_df_clean.fillna(features_df_clean.mean()).values

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    logger.info(f"Preprocessed {len(feature_cols)} features for {len(features_df)} cells")

    return features_scaled, feature_cols, scaler


def detect_isolation_forest(features_df: pd.DataFrame, contamination: float = 0.1,
                            random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    使用Isolation Forest进行异常检测

    Isolation Forest通过随机选择特征和分割值来隔离异常点，
    异常点更容易被隔离，因此路径长度更短。

    Args:
        features_df: 特征DataFrame
        contamination: 异常样本的比例（0.0-0.5之间）
        random_state: 随机种子

    Returns:
        labels: 标签数组（1表示正常，-1表示异常）
        info: 包含检测信息的字典
    """
    # 预处理特征
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # 执行Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    labels = iso_forest.fit_predict(features_scaled)

    # 获取异常分数（分数越低越异常）
    scores = iso_forest.score_samples(features_scaled)

    # 统计异常数量
    n_anomalies = (labels == -1).sum()
    n_normal = (labels == 1).sum()
    anomaly_ratio = n_anomalies / len(labels)

    info = {
        'model': iso_forest,
        'method': 'Isolation Forest',
        'contamination': contamination,
        'n_anomalies': int(n_anomalies),
        'n_normal': int(n_normal),
        'anomaly_ratio': float(anomaly_ratio),
        'scores': scores,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"Isolation Forest completed: {n_anomalies} anomalies ({anomaly_ratio:.1%}), {n_normal} normal")

    return labels, info


def detect_lof(features_df: pd.DataFrame, contamination: float = 0.1,
               n_neighbors: int = 20) -> Tuple[np.ndarray, Dict]:
    """
    使用Local Outlier Factor进行异常检测

    LOF通过比较样本的局部密度与其邻居的局部密度来检测异常，
    密度明显低于邻居的样本被认为是异常。

    Args:
        features_df: 特征DataFrame
        contamination: 异常样本的比例（0.0-0.5之间）
        n_neighbors: 邻居数量

    Returns:
        labels: 标签数组（1表示正常，-1表示异常）
        info: 包含检测信息的字典
    """
    # 预处理特征
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # 调整n_neighbors以适应样本数量
    n_samples = len(features_scaled)
    if n_neighbors >= n_samples:
        n_neighbors = max(2, n_samples // 2)
        logger.warning(f"n_neighbors too large for {n_samples} samples, adjusted to {n_neighbors}")

    # 执行LOF
    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=n_neighbors,
        novelty=False
    )
    labels = lof.fit_predict(features_scaled)

    # 获取异常分数（负值表示异常，绝对值越大越异常）
    scores = lof.negative_outlier_factor_

    # 统计异常数量
    n_anomalies = (labels == -1).sum()
    n_normal = (labels == 1).sum()
    anomaly_ratio = n_anomalies / len(labels)

    info = {
        'model': lof,
        'method': 'Local Outlier Factor',
        'contamination': contamination,
        'n_neighbors': n_neighbors,
        'n_anomalies': int(n_anomalies),
        'n_normal': int(n_normal),
        'anomaly_ratio': float(anomaly_ratio),
        'scores': scores,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"LOF completed: {n_anomalies} anomalies ({anomaly_ratio:.1%}), {n_normal} normal")

    return labels, info


def detect_one_class_svm(features_df: pd.DataFrame, nu: float = 0.1,
                         kernel: str = 'rbf', gamma: str = 'scale') -> Tuple[np.ndarray, Dict]:
    """
    使用One-Class SVM进行异常检测

    One-Class SVM学习正常数据的边界，将边界外的样本标记为异常。

    Args:
        features_df: 特征DataFrame
        nu: 异常样本的上界比例（0.0-1.0之间）
        kernel: 核函数类型（'rbf', 'linear', 'poly', 'sigmoid'）
        gamma: 核函数系数（'scale', 'auto'或浮点数）

    Returns:
        labels: 标签数组（1表示正常，-1表示异常）
        info: 包含检测信息的字典
    """
    # 预处理特征
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # 执行One-Class SVM
    oc_svm = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma=gamma
    )
    labels = oc_svm.fit_predict(features_scaled)

    # 获取决策函数值（负值表示异常）
    scores = oc_svm.decision_function(features_scaled)

    # 统计异常数量
    n_anomalies = (labels == -1).sum()
    n_normal = (labels == 1).sum()
    anomaly_ratio = n_anomalies / len(labels)

    info = {
        'model': oc_svm,
        'method': 'One-Class SVM',
        'nu': nu,
        'kernel': kernel,
        'gamma': gamma,
        'n_anomalies': int(n_anomalies),
        'n_normal': int(n_normal),
        'anomaly_ratio': float(anomaly_ratio),
        'scores': scores,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"One-Class SVM completed: {n_anomalies} anomalies ({anomaly_ratio:.1%}), {n_normal} normal")

    return labels, info


def detect_elliptic_envelope(features_df: pd.DataFrame, contamination: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    使用Elliptic Envelope进行异常检测

    假设数据服从高斯分布，通过拟合椭圆包络来检测异常。
    适用于数据近似服从多元正态分布的情况。

    Args:
        features_df: 特征DataFrame
        contamination: 异常样本的比例（0.0-0.5之间）

    Returns:
        labels: 标签数组（1表示正常，-1表示异常）
        info: 包含检测信息的字典
    """
    # 预处理特征
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # 执行Elliptic Envelope
    try:
        elliptic = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
        labels = elliptic.fit_predict(features_scaled)

        # 获取马氏距离（值越大越异常）
        scores = elliptic.decision_function(features_scaled)

        # 统计异常数量
        n_anomalies = (labels == -1).sum()
        n_normal = (labels == 1).sum()
        anomaly_ratio = n_anomalies / len(labels)

        info = {
            'model': elliptic,
            'method': 'Elliptic Envelope',
            'contamination': contamination,
            'n_anomalies': int(n_anomalies),
            'n_normal': int(n_normal),
            'anomaly_ratio': float(anomaly_ratio),
            'scores': scores,
            'feature_cols': feature_cols,
            'scaler': scaler
        }

        logger.info(f"Elliptic Envelope completed: {n_anomalies} anomalies ({anomaly_ratio:.1%}), {n_normal} normal")

    except Exception as e:
        logger.error(f"Elliptic Envelope failed: {str(e)}")
        # 如果失败，返回全部为正常的标签
        labels = np.ones(len(features_scaled), dtype=int)
        info = {
            'model': None,
            'method': 'Elliptic Envelope',
            'contamination': contamination,
            'n_anomalies': 0,
            'n_normal': len(labels),
            'anomaly_ratio': 0.0,
            'scores': np.zeros(len(labels)),
            'feature_cols': feature_cols,
            'scaler': scaler,
            'error': str(e)
        }

    return labels, info


def get_anomaly_statistics(features_df: pd.DataFrame, labels: np.ndarray,
                           exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    获取异常样本的统计信息

    Args:
        features_df: 特征DataFrame
        labels: 异常检测标签（1表示正常，-1表示异常）
        exclude_cols: 需要排除的列名列表

    Returns:
        统计信息DataFrame
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

    # 分离正常和异常样本
    normal_mask = labels == 1
    anomaly_mask = labels == -1

    stats_list = []

    for col in feature_cols:
        normal_values = features_df.loc[normal_mask, col]
        anomaly_values = features_df.loc[anomaly_mask, col]

        stats = {
            'feature': col,
            'normal_mean': normal_values.mean() if len(normal_values) > 0 else 0,
            'normal_std': normal_values.std() if len(normal_values) > 0 else 0,
            'anomaly_mean': anomaly_values.mean() if len(anomaly_values) > 0 else 0,
            'anomaly_std': anomaly_values.std() if len(anomaly_values) > 0 else 0,
            'difference': abs(normal_values.mean() - anomaly_values.mean()) if len(normal_values) > 0 and len(anomaly_values) > 0 else 0
        }

        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df.sort_values('difference', ascending=False).reset_index(drop=True)

    logger.info(f"Computed anomaly statistics for {len(feature_cols)} features")

    return stats_df


def _reduce_to_2d(features_scaled: np.ndarray) -> np.ndarray:
    """
    使用PCA将特征降至2D用于可视化

    Args:
        features_scaled: 标准化后的特征数组

    Returns:
        2D特征数组
    """
    if features_scaled.shape[1] <= 2:
        return features_scaled

    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features_scaled)

    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    return features_2d


def visualize_isolation_forest(features_scaled: np.ndarray, labels: np.ndarray,
                                scores: np.ndarray, info: Dict) -> plt.Figure:
    """
    可视化Isolation Forest异常检测结果 - 展示空间分割过程

    Isolation Forest通过随机选择特征和分割点来递归分割空间，
    本可视化展示了这种空间分割的过程和异常分数的分布。

    Args:
        features_scaled: 标准化后的特征数组
        labels: 异常检测标签（1表示正常，-1表示异常）
        scores: 异常分数（分数越低越异常）
        info: 检测信息字典

    Returns:
        matplotlib Figure对象
    """
    # 降维到2D用于可视化
    features_2d = _reduce_to_2d(features_scaled)

    # 创建图形 - 提高分辨率和尺寸（DPI提升到300以获得更高清晰度）
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), dpi=300)
    fig.suptitle('Isolation Forest: Space Partitioning Visualization', fontsize=20, fontweight='bold')

    normal_mask = labels == 1
    anomaly_mask = labels == -1

    # 1. 空间分割可视化 - 展示Isolation Forest如何分割空间
    ax1 = axes[0, 0]

    # 创建网格用于显示决策边界
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # 使用模型预测网格点的异常分数
    model = info['model']
    # 需要将2D点映射回原始特征空间
    if features_scaled.shape[1] > 2:
        # 使用PCA逆变换（近似）
        pca = PCA(n_components=2, random_state=42)
        pca.fit(features_scaled)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        # 这里我们直接在2D空间上可视化，使用2D特征训练一个新的IF模型
        if features_scaled.shape[1] > 2:
            # 使用前2个主成分重新训练模型用于可视化
            iso_forest_2d = IsolationForest(
                contamination=info['contamination'],
                random_state=42,
                n_estimators=100
            )
            iso_forest_2d.fit(features_2d)
            Z = iso_forest_2d.score_samples(grid_points)
        else:
            Z = model.score_samples(grid_points)
    else:
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.score_samples(grid_points)

    Z = Z.reshape(xx.shape)

    # 绘制异常分数的等高线（展示空间分割）
    contour = ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.7)
    ax1.contour(xx, yy, Z, levels=[scores[anomaly_mask].max()],
                colors='red', linewidths=4, linestyles='dashed', label='Decision Boundary')

    # 绘制样本点 - 增大尺寸提高可见度
    ax1.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1],
                c='blue', alpha=0.7, s=60, label='Normal', edgecolors='k', linewidth=1.0)
    ax1.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                c='red', alpha=0.9, s=120, label='Anomaly', edgecolors='k', linewidth=1.5, marker='^')

    ax1.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax1.set_title('Space Partitioning by Isolation Forest', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    plt.colorbar(contour, ax=ax1, label='Anomaly Score')
    ax1.grid(True, alpha=0.2)

    # 2. 路径长度分布 - 展示异常点的隔离路径更短
    ax2 = axes[0, 1]

    # 异常分数越低，路径长度越短（越容易被隔离）
    # 将分数转换为路径长度的概念（反向）
    path_lengths = -scores  # 分数越低，路径越短

    ax2.hist(path_lengths[normal_mask], bins=30, alpha=0.6, color='blue',
             label=f'Normal (mean={path_lengths[normal_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.hist(path_lengths[anomaly_mask], bins=30, alpha=0.6, color='red',
             label=f'Anomaly (mean={path_lengths[anomaly_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.axvline(x=path_lengths[normal_mask].mean(), color='blue', linestyle='--', linewidth=3)
    ax2.axvline(x=path_lengths[anomaly_mask].mean(), color='red', linestyle='--', linewidth=3)

    ax2.set_xlabel('Path Length (shorter = more anomalous)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax2.set_title('Isolation Path Length Distribution', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加说明文本 - 增大字体
    ax2.text(0.02, 0.98, 'Anomalies have shorter\nisolation paths',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 3. 树的分割示意图 - 展示随机分割过程
    ax3 = axes[1, 0]

    # 选择几个代表性的分割线来展示
    # 模拟随机分割过程
    np.random.seed(42)
    n_splits = 8  # 显示8条分割线

    # 绘制背景 - 增大样本点
    ax3.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1],
                c='lightblue', alpha=0.4, s=50, edgecolors='none')
    ax3.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                c='lightcoral', alpha=0.6, s=90, marker='^', edgecolors='none')

    # 绘制随机分割线 - 增加粗细
    colors = plt.cm.Set3(np.linspace(0, 1, n_splits))
    for i in range(n_splits):
        # 随机选择分割方向（水平或垂直）
        if np.random.rand() > 0.5:
            # 垂直分割
            split_pos = np.random.uniform(x_min, x_max)
            ax3.axvline(x=split_pos, color=colors[i], linestyle='-', linewidth=3, alpha=0.8)
        else:
            # 水平分割
            split_pos = np.random.uniform(y_min, y_max)
            ax3.axhline(y=split_pos, color=colors[i], linestyle='-', linewidth=3, alpha=0.8)

    ax3.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax3.set_title('Random Space Partitioning (Sample Splits)', fontsize=15, fontweight='bold')
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.grid(True, alpha=0.2)

    # 添加说明 - 增大字体
    ax3.text(0.02, 0.98, 'Colored lines show\nrandom splits',
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # 4. 统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    Algorithm: Isolation Forest

    How it works:
    • Randomly selects features and split values
    • Recursively partitions the space
    • Anomalies are isolated faster (shorter paths)
    • Creates an ensemble of isolation trees

    Detection Results:
    • Total Samples: {len(labels)}
    • Normal: {info['n_normal']} ({info['n_normal']/len(labels)*100:.1f}%)
    • Anomaly: {info['n_anomalies']} ({info['anomaly_ratio']*100:.1f}%)

    Parameters:
    • Contamination: {info['contamination']}
    • Number of Trees: 100

    Anomaly Score Statistics:
    • Normal mean: {scores[normal_mask].mean():.4f}
    • Anomaly mean: {scores[anomaly_mask].mean():.4f}
    • Threshold: {scores[anomaly_mask].max():.4f}

    Interpretation:
    • Lower scores indicate anomalies
    • Anomalies require fewer splits to isolate
    • Red dashed line shows decision boundary
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    logger.info("Isolation Forest space partitioning visualization created")

    return fig


def visualize_lof(features_scaled: np.ndarray, labels: np.ndarray,
                  scores: np.ndarray, info: Dict) -> plt.Figure:
    """
    可视化Local Outlier Factor异常检测结果 - 展示局部密度概念

    LOF通过比较样本的局部密度与其邻居的局部密度来检测异常，
    本可视化展示局部密度、k-近邻关系和密度对比。

    Args:
        features_scaled: 标准化后的特征数组
        labels: 异常检测标签（1表示正常，-1表示异常）
        scores: LOF分数（负值表示异常，绝对值越大越异常）
        info: 检测信息字典

    Returns:
        matplotlib Figure对象
    """
    # 降维到2D用于可视化
    features_2d = _reduce_to_2d(features_scaled)

    # 创建图形 - 提高分辨率和尺寸（DPI提升到300以获得更高清晰度）
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), dpi=300)
    fig.suptitle('Local Outlier Factor (LOF): Local Density Visualization', fontsize=20, fontweight='bold')

    normal_mask = labels == 1
    anomaly_mask = labels == -1

    # 1. 局部密度可视化 - 展示LOF如何基于局部密度检测异常
    ax1 = axes[0, 0]

    # 使用LOF分数的绝对值表示局部密度（分数越负，密度越低）
    # 将LOF分数转换为密度指标（越正常，密度越高）
    density_scores = -scores  # 反转，使得正常样本有更高的值

    # 绘制密度热图 - 增大样本点
    scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=density_scores, cmap='RdYlGn_r', alpha=0.7, s=100,
                          edgecolors='k', linewidth=1.0)

    # 标记异常点 - 增大尺寸
    ax1.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                facecolors='none', edgecolors='red', linewidths=4, s=200,
                marker='o', label='Anomaly (Low Density)')

    ax1.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax1.set_title('Local Density Distribution', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Local Density (higher = denser)', fontsize=12)
    ax1.grid(True, alpha=0.2)

    # 2. LOF分数分布
    ax2 = axes[0, 1]

    ax2.hist(scores[normal_mask], bins=30, alpha=0.6, color='blue',
             label=f'Normal (mean={scores[normal_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.hist(scores[anomaly_mask], bins=30, alpha=0.6, color='red',
             label=f'Anomaly (mean={scores[anomaly_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.axvline(x=scores[normal_mask].mean(), color='blue', linestyle='--', linewidth=3)
    ax2.axvline(x=scores[anomaly_mask].mean(), color='red', linestyle='--', linewidth=3)

    ax2.set_xlabel('LOF Score (more negative = more anomalous)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax2.set_title('LOF Score Distribution', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加说明
    ax2.text(0.02, 0.98, 'Lower density regions\nhave more negative scores',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 3. k-近邻关系可视化 - 展示局部邻居关系
    ax3 = axes[1, 0]

    # 绘制所有样本 - 增大尺寸
    ax3.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1],
                c='lightblue', alpha=0.5, s=60, edgecolors='k', linewidth=0.7)
    ax3.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                c='lightcoral', alpha=0.7, s=100, marker='^', edgecolors='k', linewidth=1.0)

    # 选择几个代表性的异常点，展示其k-近邻
    from sklearn.neighbors import NearestNeighbors
    n_neighbors = min(info['n_neighbors'], len(features_2d) - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(features_2d)

    # 选择最异常的3个点展示邻居关系
    anomaly_indices = np.where(anomaly_mask)[0]
    if len(anomaly_indices) > 0:
        # 选择LOF分数最低的几个异常点
        top_anomalies = anomaly_indices[np.argsort(scores[anomaly_indices])[:min(3, len(anomaly_indices))]]

        colors_neighbors = ['red', 'orange', 'purple']
        for idx, anomaly_idx in enumerate(top_anomalies):
            # 找到k-近邻
            distances, indices = nbrs.kneighbors([features_2d[anomaly_idx]])

            # 绘制连线 - 增加粗细
            for neighbor_idx in indices[0][1:]:  # 跳过自己
                ax3.plot([features_2d[anomaly_idx, 0], features_2d[neighbor_idx, 0]],
                        [features_2d[anomaly_idx, 1], features_2d[neighbor_idx, 1]],
                        color=colors_neighbors[idx], alpha=0.5, linewidth=2.0)

            # 高亮异常点 - 增大尺寸
            ax3.scatter(features_2d[anomaly_idx, 0], features_2d[anomaly_idx, 1],
                       c=colors_neighbors[idx], s=300, marker='*', edgecolors='black',
                       linewidth=2.5, zorder=10, label=f'Anomaly {idx+1}')

    ax3.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax3.set_title(f'k-Nearest Neighbors (k={n_neighbors})', fontsize=15, fontweight='bold')
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.2)

    # 添加说明
    ax3.text(0.02, 0.98, 'Lines show k-nearest\nneighbors of anomalies',
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # 4. 统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    Algorithm: Local Outlier Factor (LOF)

    How it works:
    • Compares local density of each sample
    • Density based on k-nearest neighbors
    • Low density samples are anomalies
    • Considers local neighborhood structure

    Detection Results:
    • Total Samples: {len(labels)}
    • Normal: {info['n_normal']} ({info['n_normal']/len(labels)*100:.1f}%)
    • Anomaly: {info['n_anomalies']} ({info['anomaly_ratio']*100:.1f}%)

    Parameters:
    • Contamination: {info['contamination']}
    • Number of Neighbors: {info['n_neighbors']}

    LOF Score Statistics:
    • Normal mean: {scores[normal_mask].mean():.4f}
    • Anomaly mean: {scores[anomaly_mask].mean():.4f}
    • Threshold: {scores[anomaly_mask].min():.4f}

    Interpretation:
    • Scores close to -1.0 = normal density
    • More negative scores = lower local density
    • Anomalies have fewer nearby neighbors
    • Lines show k-nearest neighbor connections
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    logger.info("LOF local density visualization created")

    return fig


def visualize_one_class_svm(features_scaled: np.ndarray, labels: np.ndarray,
                             scores: np.ndarray, info: Dict) -> plt.Figure:
    """
    可视化One-Class SVM异常检测结果 - 展示决策边界和核函数效果

    One-Class SVM学习正常数据的边界，使用核函数将数据映射到高维空间，
    本可视化展示决策边界、支持向量和核函数的效果。

    Args:
        features_scaled: 标准化后的特征数组
        labels: 异常检测标签（1表示正常，-1表示异常）
        scores: 决策函数值（负值表示异常）
        info: 检测信息字典

    Returns:
        matplotlib Figure对象
    """
    # 降维到2D用于可视化
    features_2d = _reduce_to_2d(features_scaled)

    # 创建图形 - 提高分辨率和尺寸（DPI提升到300以获得更高清晰度）
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), dpi=300)
    fig.suptitle('One-Class SVM: Decision Boundary Visualization', fontsize=20, fontweight='bold')

    normal_mask = labels == 1
    anomaly_mask = labels == -1

    # 1. 决策边界可视化 - 展示SVM如何学习正常数据的边界
    ax1 = axes[0, 0]

    # 创建网格用于显示决策边界
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # 在2D空间上训练One-Class SVM用于可视化
    from sklearn.svm import OneClassSVM
    svm_2d = OneClassSVM(
        nu=info['nu'],
        kernel=info['kernel'],
        gamma=info['gamma']
    )
    svm_2d.fit(features_2d)

    # 计算网格点的决策函数值
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_2d.decision_function(grid_points)
    Z = Z.reshape(xx.shape)

    # 绘制决策边界的等高线 - 增加层次
    contour = ax1.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 25),
                           cmap='RdYlBu', alpha=0.7)
    # 绘制决策边界（decision_function = 0）- 增加粗细
    ax1.contour(xx, yy, Z, levels=[0], colors='red', linewidths=4,
                linestyles='solid', label='Decision Boundary')
    # 绘制边界区域（margin）- 增加粗细
    ax1.contour(xx, yy, Z, levels=[-0.5, 0.5], colors='orange',
                linewidths=3, linestyles='dashed', alpha=0.8)

    # 绘制样本点 - 增大尺寸
    ax1.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1],
                c='blue', alpha=0.7, s=60, label='Normal', edgecolors='k', linewidth=1.0)
    ax1.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                c='red', alpha=0.9, s=120, label='Anomaly', edgecolors='k', linewidth=1.5, marker='^')

    # 标记支持向量 - 增大尺寸
    support_vectors_2d = features_2d[svm_2d.support_]
    ax1.scatter(support_vectors_2d[:, 0], support_vectors_2d[:, 1],
                s=250, facecolors='none', edgecolors='green', linewidths=3.5,
                label='Support Vectors')

    ax1.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Decision Boundary ({info["kernel"]} kernel)', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    plt.colorbar(contour, ax=ax1, label='Decision Function Value')
    ax1.grid(True, alpha=0.2)

    # 2. 决策函数值分布
    ax2 = axes[0, 1]

    ax2.hist(scores[normal_mask], bins=30, alpha=0.6, color='blue',
             label=f'Normal (mean={scores[normal_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.hist(scores[anomaly_mask], bins=30, alpha=0.6, color='red',
             label=f'Anomaly (mean={scores[anomaly_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=4, label='Decision Boundary (0)')
    ax2.axvline(x=scores[normal_mask].mean(), color='blue', linestyle=':', linewidth=3)
    ax2.axvline(x=scores[anomaly_mask].mean(), color='red', linestyle=':', linewidth=3)

    ax2.set_xlabel('Decision Function Value', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax2.set_title('Decision Function Distribution', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加说明
    ax2.text(0.02, 0.98, 'Positive = Normal\nNegative = Anomaly',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 3. 核函数效果展示 - 距离到决策边界的映射
    ax3 = axes[1, 0]

    # 绘制决策函数值的3D效果（使用颜色表示）- 增大样本点
    scatter = ax3.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=scores, cmap='RdYlBu', alpha=0.8, s=100,
                          edgecolors='k', linewidth=1.0)

    # 绘制决策边界 - 增加粗细
    ax3.contour(xx, yy, Z, levels=[0], colors='red', linewidths=4, linestyles='solid')

    # 标记支持向量 - 增大尺寸
    ax3.scatter(support_vectors_2d[:, 0], support_vectors_2d[:, 1],
                s=250, facecolors='none', edgecolors='green', linewidths=3.5)

    ax3.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax3.set_title('Kernel Function Effect', fontsize=15, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Distance to Boundary', fontsize=12)
    ax3.grid(True, alpha=0.2)

    # 添加说明 - 增大字体
    ax3.text(0.02, 0.98, f'Kernel: {info["kernel"]}\nSupport Vectors: {len(svm_2d.support_)}',
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 4. 统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    Algorithm: One-Class SVM

    How it works:
    • Learns a boundary around normal data
    • Uses kernel trick to map to high-dimensional space
    • Maximizes margin from origin in feature space
    • Support vectors define the boundary

    Detection Results:
    • Total Samples: {len(labels)}
    • Normal: {info['n_normal']} ({info['n_normal']/len(labels)*100:.1f}%)
    • Anomaly: {info['n_anomalies']} ({info['anomaly_ratio']*100:.1f}%)
    • Support Vectors: {len(svm_2d.support_)} ({len(svm_2d.support_)/len(labels)*100:.1f}%)

    Parameters:
    • Nu: {info['nu']} (upper bound on anomaly fraction)
    • Kernel: {info['kernel']}
    • Gamma: {info['gamma']}

    Decision Function Statistics:
    • Normal mean: {scores[normal_mask].mean():.4f}
    • Anomaly mean: {scores[anomaly_mask].mean():.4f}
    • Boundary: 0.0

    Interpretation:
    • Positive values = inside boundary (normal)
    • Negative values = outside boundary (anomaly)
    • Green circles = support vectors
    • Red line = decision boundary
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    logger.info("One-Class SVM decision boundary visualization created")

    return fig


def visualize_elliptic_envelope(features_scaled: np.ndarray, labels: np.ndarray,
                                  scores: np.ndarray, info: Dict) -> plt.Figure:
    """
    可视化Elliptic Envelope异常检测结果 - 展示椭圆包络边界

    Elliptic Envelope假设数据服从高斯分布，通过拟合椭圆包络来检测异常，
    本可视化展示椭圆边界、马氏距离和高斯分布假设。

    Args:
        features_scaled: 标准化后的特征数组
        labels: 异常检测标签（1表示正常，-1表示异常）
        scores: 马氏距离（值越大越异常）
        info: 检测信息字典

    Returns:
        matplotlib Figure对象
    """
    # 降维到2D用于可视化
    features_2d = _reduce_to_2d(features_scaled)

    # 创建图形 - 提高分辨率和尺寸（DPI提升到300以获得更高清晰度）
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), dpi=300)
    fig.suptitle('Elliptic Envelope: Gaussian Distribution Boundary', fontsize=20, fontweight='bold')

    normal_mask = labels == 1
    anomaly_mask = labels == -1

    # 1. 椭圆包络边界可视化 - 展示高斯分布的边界
    ax1 = axes[0, 0]

    # 创建网格用于显示决策边界
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # 在2D空间上训练Elliptic Envelope用于可视化
    from sklearn.covariance import EllipticEnvelope
    elliptic_2d = EllipticEnvelope(
        contamination=info['contamination'],
        random_state=42
    )
    elliptic_2d.fit(features_2d)

    # 计算网格点的马氏距离
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = elliptic_2d.decision_function(grid_points)
    Z = Z.reshape(xx.shape)

    # 绘制马氏距离的等高线（展示椭圆包络）- 增加层次
    contour = ax1.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 25),
                           cmap='RdYlBu', alpha=0.7)
    # 绘制椭圆边界（decision_function = 0）- 增加粗细
    ax1.contour(xx, yy, Z, levels=[0], colors='red', linewidths=4,
                linestyles='solid', label='Elliptic Boundary')
    # 绘制置信区间 - 增加粗细
    ax1.contour(xx, yy, Z, levels=[-2, -1, 1, 2], colors='orange',
                linewidths=2.5, linestyles='dashed', alpha=0.8)

    # 绘制样本点 - 增大尺寸
    ax1.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1],
                c='blue', alpha=0.7, s=60, label='Normal', edgecolors='k', linewidth=1.0)
    ax1.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                c='red', alpha=0.9, s=120, label='Anomaly', edgecolors='k', linewidth=1.5, marker='^')

    # 标记分布中心 - 增大尺寸
    center = elliptic_2d.location_
    ax1.scatter(center[0], center[1], c='green', s=400, marker='X',
                edgecolors='black', linewidths=2.5, label='Distribution Center', zorder=10)

    ax1.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax1.set_title('Elliptic Envelope Boundary', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    plt.colorbar(contour, ax=ax1, label='Mahalanobis Distance')
    ax1.grid(True, alpha=0.2)

    # 2. 马氏距离分布
    ax2 = axes[0, 1]

    ax2.hist(scores[normal_mask], bins=30, alpha=0.6, color='blue',
             label=f'Normal (mean={scores[normal_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.hist(scores[anomaly_mask], bins=30, alpha=0.6, color='red',
             label=f'Anomaly (mean={scores[anomaly_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=4, label='Boundary (0)')
    ax2.axvline(x=scores[normal_mask].mean(), color='blue', linestyle=':', linewidth=3)
    ax2.axvline(x=scores[anomaly_mask].mean(), color='red', linestyle=':', linewidth=3)

    ax2.set_xlabel('Mahalanobis Distance', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax2.set_title('Mahalanobis Distance Distribution', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加说明 - 增大字体
    ax2.text(0.02, 0.98, 'Positive = Inside ellipse\nNegative = Outside ellipse',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 3. 协方差椭圆可视化 - 展示高斯分布的形状
    ax3 = axes[1, 0]

    # 绘制马氏距离的热图 - 增大样本点
    scatter = ax3.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=scores, cmap='RdYlBu', alpha=0.8, s=100,
                          edgecolors='k', linewidth=1.0)

    # 绘制椭圆边界 - 增加粗细
    ax3.contour(xx, yy, Z, levels=[0], colors='red', linewidths=4, linestyles='solid')
    # 绘制多个置信椭圆 - 增加粗细
    ax3.contour(xx, yy, Z, levels=[-2, -1, 1, 2], colors='orange',
                linewidths=2.5, linestyles='dashed', alpha=0.8)

    # 标记分布中心 - 增大尺寸
    ax3.scatter(center[0], center[1], c='green', s=400, marker='X',
                edgecolors='black', linewidths=2.5, zorder=10)

    ax3.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax3.set_title('Covariance Ellipse (Gaussian Assumption)', fontsize=15, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Mahalanobis Distance', fontsize=12)
    ax3.grid(True, alpha=0.2)

    # 添加说明 - 增大字体
    ax3.text(0.02, 0.98, 'Dashed lines:\nConfidence intervals',
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 4. 统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    Algorithm: Elliptic Envelope

    How it works:
    • Assumes data follows Gaussian distribution
    • Fits an ellipse around normal data
    • Uses Mahalanobis distance to measure outliers
    • Robust covariance estimation

    Detection Results:
    • Total Samples: {len(labels)}
    • Normal: {info['n_normal']} ({info['n_normal']/len(labels)*100:.1f}%)
    • Anomaly: {info['n_anomalies']} ({info['anomaly_ratio']*100:.1f}%)

    Parameters:
    • Contamination: {info['contamination']}

    Mahalanobis Distance Statistics:
    • Normal mean: {scores[normal_mask].mean():.4f}
    • Anomaly mean: {scores[anomaly_mask].mean():.4f}
    • Boundary: 0.0

    Interpretation:
    • Positive values = inside ellipse (normal)
    • Negative values = outside ellipse (anomaly)
    • Green X = distribution center
    • Red line = elliptic boundary
    • Orange dashed = confidence intervals
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    logger.info("Elliptic Envelope boundary visualization created")

    return fig
