"""
虚拟筛选模块

使用训练好的监督学习模型对新数据进行批量预测和筛选
支持置信度评分、结果排序和候选物筛选

主要功能：
1. 加载训练好的模型进行预测
2. 计算预测置信度
3. 结果排序和过滤
4. 可视化筛选结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import joblib
from loguru import logger

# Sklearn imports
from sklearn.metrics import r2_score

# ============================================================================
# CORE SCREENING FUNCTIONS
# ============================================================================

def screen_dataset(
    model_path: str,
    data_df: pd.DataFrame,
    confidence_method: str = 'probability',
    min_confidence: Optional[float] = None,
    return_probabilities: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    使用训练好的模型对数据集进行虚拟筛选

    Args:
        model_path: 模型文件路径
        data_df: 待筛选的特征DataFrame
        confidence_method: 置信度计算方法 ('probability', 'distance')
        min_confidence: 最小置信度阈值（None表示不过滤）
        return_probabilities: 是否返回概率（分类任务）

    Returns:
        results_df: 包含预测结果和置信度的DataFrame
        info: 筛选统计信息字典
    """
    logger.info(f"Loading model from {model_path}")

    # 加载模型
    try:
        model_package = joblib.load(model_path)
        model = model_package['model']
        task_type = model_package.get('task_type', 'classification')
        feature_names = model_package.get('feature_names', [])
        scaler = model_package.get('scaler')
        encoder = model_package.get('encoder')
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    logger.info(f"Model loaded: task_type={task_type}, n_features={len(feature_names)}")

    # 验证特征
    missing_features = [f for f in feature_names if f not in data_df.columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        raise ValueError(f"Missing features in data: {missing_features}")

    # 提取特征
    X = data_df[feature_names]

    # 应用缩放器
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    # 预测
    logger.info(f"Screening {len(data_df)} samples...")
    predictions = model.predict(X_scaled)

    # 创建结果DataFrame
    results_df = data_df.copy()
    results_df['prediction'] = predictions

    # 计算置信度
    if task_type == 'classification':
        if hasattr(model, 'predict_proba') and return_probabilities:
            probabilities = model.predict_proba(X_scaled)

            # 添加每个类别的概率
            for i in range(probabilities.shape[1]):
                results_df[f'probability_class_{i}'] = probabilities[:, i]

            # 置信度为最大概率
            if confidence_method == 'probability':
                confidence = np.max(probabilities, axis=1)
            else:
                confidence = np.max(probabilities, axis=1)
        else:
            confidence = np.ones(len(predictions))
    else:
        # 回归任务：置信度设为1（或可以基于预测区间计算）
        confidence = np.ones(len(predictions))

    results_df['confidence'] = confidence

    # 过滤低置信度样本
    if min_confidence is not None:
        n_before = len(results_df)
        results_df = results_df[results_df['confidence'] >= min_confidence]
        n_after = len(results_df)
        logger.info(f"Filtered by confidence: {n_before} -> {n_after} samples")

    # 统计信息
    info = {
        'n_samples': len(data_df),
        'n_screened': len(results_df),
        'task_type': task_type,
        'model_path': model_path,
        'confidence_method': confidence_method,
        'min_confidence': min_confidence,
        'prediction_stats': {
            'mean': float(predictions.mean()),
            'std': float(predictions.std()),
            'min': float(predictions.min()),
            'max': float(predictions.max())
        }
    }

    if task_type == 'classification':
        unique, counts = np.unique(predictions, return_counts=True)
        info['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))

    logger.info(f"Screening completed: {len(results_df)} samples")

    return results_df, info


def batch_screen_files(
    model_path: str,
    data_files: List[str],
    output_dir: str,
    confidence_threshold: float = 0.7,
    merge_results: bool = True
) -> Optional[pd.DataFrame]:
    """
    批量筛选多个CSV文件

    Args:
        model_path: 模型文件路径
        data_files: 数据文件路径列表
        output_dir: 输出目录
        confidence_threshold: 最小置信度阈值
        merge_results: 是否合并所有结果

    Returns:
        merged_results_df: 合并的结果DataFrame（如果merge_results=True）
    """
    logger.info(f"Batch screening {len(data_files)} files")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, data_file in enumerate(data_files):
        try:
            logger.info(f"Processing file {i+1}/{len(data_files)}: {data_file}")

            # 读取数据
            data_df = pd.read_csv(data_file)

            # 筛选
            results_df, info = screen_dataset(
                model_path, data_df,
                min_confidence=confidence_threshold
            )

            # 保存结果
            output_file = Path(output_dir) / f"screened_{Path(data_file).stem}.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")

            if merge_results:
                all_results.append(results_df)

        except Exception as e:
            logger.error(f"Failed to process {data_file}: {str(e)}")
            continue

    # 合并结果
    if merge_results and all_results:
        merged_df = pd.concat(all_results, ignore_index=True)
        merged_file = Path(output_dir) / "merged_results.csv"
        merged_df.to_csv(merged_file, index=False)
        logger.info(f"Merged results saved to {merged_file}")
        return merged_df

    return None


# ============================================================================
# CONFIDENCE SCORING FUNCTIONS
# ============================================================================

def compute_confidence_probability(
    model: Any,
    X: np.ndarray
) -> np.ndarray:
    """
    基于概率的置信度计算（分类任务）

    Args:
        model: 训练好的模型
        X: 特征数组

    Returns:
        confidence_scores: 置信度分数数组
    """
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        # 置信度为最大概率
        confidence = np.max(probabilities, axis=1)
    else:
        # 如果模型不支持概率预测，返回全1
        confidence = np.ones(len(X))

    return confidence


def compute_prediction_intervals(
    model: Any,
    X: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算预测区间（回归任务）

    使用简单的方法基于训练误差估计预测区间

    Args:
        model: 训练好的模型
        X: 特征数组
        confidence_level: 置信水平

    Returns:
        lower_bounds: 下界数组
        upper_bounds: 上界数组
    """
    predictions = model.predict(X)

    # 简单方法：假设误差为正态分布
    # 这里使用固定的标准差估计（实际应该从训练数据计算）
    # 更准确的方法需要保存训练时的残差信息
    std_estimate = predictions.std() * 0.1  # 简化估计

    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    lower_bounds = predictions - z_score * std_estimate
    upper_bounds = predictions + z_score * std_estimate

    return lower_bounds, upper_bounds


# ============================================================================
# RESULT RANKING FUNCTIONS
# ============================================================================

def rank_by_prediction(
    results_df: pd.DataFrame,
    prediction_col: str = 'prediction',
    confidence_col: str = 'confidence',
    ascending: bool = False
) -> pd.DataFrame:
    """
    根据预测值排序

    Args:
        results_df: 结果DataFrame
        prediction_col: 预测列名
        confidence_col: 置信度列名
        ascending: 是否升序

    Returns:
        ranked_df: 排序后的DataFrame
    """
    # 先按预测值排序，再按置信度排序
    ranked_df = results_df.sort_values(
        by=[prediction_col, confidence_col],
        ascending=[ascending, False]
    ).reset_index(drop=True)

    return ranked_df


def filter_by_confidence(
    results_df: pd.DataFrame,
    confidence_col: str = 'confidence',
    min_confidence: float = 0.7
) -> pd.DataFrame:
    """
    根据置信度过滤

    Args:
        results_df: 结果DataFrame
        confidence_col: 置信度列名
        min_confidence: 最小置信度阈值

    Returns:
        filtered_df: 过滤后的DataFrame
    """
    filtered_df = results_df[results_df[confidence_col] >= min_confidence].copy()
    logger.info(f"Filtered by confidence >= {min_confidence}: {len(results_df)} -> {len(filtered_df)} samples")
    return filtered_df


def filter_by_prediction_range(
    results_df: pd.DataFrame,
    prediction_col: str = 'prediction',
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> pd.DataFrame:
    """
    根据预测值范围过滤

    Args:
        results_df: 结果DataFrame
        prediction_col: 预测列名
        min_value: 最小值
        max_value: 最大值

    Returns:
        filtered_df: 过滤后的DataFrame
    """
    filtered_df = results_df.copy()

    if min_value is not None:
        filtered_df = filtered_df[filtered_df[prediction_col] >= min_value]

    if max_value is not None:
        filtered_df = filtered_df[filtered_df[prediction_col] <= max_value]

    logger.info(f"Filtered by prediction range [{min_value}, {max_value}]: {len(results_df)} -> {len(filtered_df)} samples")

    return filtered_df


def select_top_candidates(
    results_df: pd.DataFrame,
    n_candidates: int = 100,
    criteria: str = 'prediction',
    confidence_threshold: float = 0.5,
    prediction_col: str = 'prediction',
    confidence_col: str = 'confidence'
) -> pd.DataFrame:
    """
    选择顶部候选物

    Args:
        results_df: 结果DataFrame
        n_candidates: 候选物数量
        criteria: 'prediction', 'confidence', 'combined'
        confidence_threshold: 最小置信度阈值
        prediction_col: 预测列名
        confidence_col: 置信度列名

    Returns:
        top_candidates_df: 顶部候选物DataFrame
    """
    # 先过滤置信度
    filtered_df = results_df[results_df[confidence_col] >= confidence_threshold].copy()

    if len(filtered_df) == 0:
        logger.warning(f"No samples meet confidence threshold {confidence_threshold}")
        return pd.DataFrame()

    # 根据标准排序
    if criteria == 'prediction':
        sorted_df = filtered_df.sort_values(prediction_col, ascending=False)
    elif criteria == 'confidence':
        sorted_df = filtered_df.sort_values(confidence_col, ascending=False)
    elif criteria == 'combined':
        # 组合分数：预测值 * 置信度
        filtered_df['combined_score'] = filtered_df[prediction_col] * filtered_df[confidence_col]
        sorted_df = filtered_df.sort_values('combined_score', ascending=False)
    else:
        raise ValueError(f"Unknown criteria: {criteria}")

    # 选择前N个
    top_candidates = sorted_df.head(n_candidates)

    logger.info(f"Selected top {len(top_candidates)} candidates (criteria={criteria})")

    return top_candidates


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_prediction_distribution(
    results_df: pd.DataFrame,
    prediction_col: str = 'prediction',
    task_type: str = 'classification'
) -> plt.Figure:
    """
    绘制预测分布直方图

    Args:
        results_df: 结果DataFrame
        prediction_col: 预测列名
        task_type: 'classification' or 'regression'

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    predictions = results_df[prediction_col]

    if task_type == 'classification':
        # 分类任务：条形图
        unique, counts = np.unique(predictions, return_counts=True)
        ax.bar(unique, counts, color='steelblue', edgecolor='black')
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        # 回归任务：直方图
        ax.hist(predictions, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Predicted Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_confidence_distribution(
    results_df: pd.DataFrame,
    confidence_col: str = 'confidence'
) -> plt.Figure:
    """
    绘制置信度分布

    Args:
        results_df: 结果DataFrame
        confidence_col: 置信度列名

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    confidence = results_df[confidence_col]

    ax.hist(confidence, bins=30, color='green', edgecolor='black', alpha=0.7)
    ax.axvline(confidence.mean(), color='r', linestyle='--', lw=2,
               label=f'Mean = {confidence.mean():.3f}')
    ax.axvline(confidence.median(), color='orange', linestyle='--', lw=2,
               label=f'Median = {confidence.median():.3f}')

    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_top_candidates(
    results_df: pd.DataFrame,
    top_n: int = 20,
    prediction_col: str = 'prediction',
    confidence_col: str = 'confidence'
) -> plt.Figure:
    """
    绘制顶部候选物可视化

    Args:
        results_df: 结果DataFrame
        top_n: 显示前N个候选物
        prediction_col: 预测列名
        confidence_col: 置信度列名

    Returns:
        fig: matplotlib Figure对象
    """
    # 选择前N个
    top_df = results_df.head(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # 预测值条形图
    x_pos = np.arange(len(top_df))
    ax1.bar(x_pos, top_df[prediction_col], color='steelblue', edgecolor='black')
    ax1.set_xlabel('Candidate Rank', fontsize=12)
    ax1.set_ylabel('Predicted Value', fontsize=12)
    ax1.set_title(f'Top {top_n} Candidates - Predictions', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 置信度条形图
    colors = ['green' if c >= 0.7 else 'orange' if c >= 0.5 else 'red'
              for c in top_df[confidence_col]]
    ax2.bar(x_pos, top_df[confidence_col], color=colors, edgecolor='black')
    ax2.axhline(y=0.7, color='green', linestyle='--', lw=2, alpha=0.5, label='High Confidence')
    ax2.axhline(y=0.5, color='orange', linestyle='--', lw=2, alpha=0.5, label='Medium Confidence')
    ax2.set_xlabel('Candidate Rank', fontsize=12)
    ax2.set_ylabel('Confidence Score', fontsize=12)
    ax2.set_title(f'Top {top_n} Candidates - Confidence', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_confidence_intervals(
    results_df: pd.DataFrame,
    prediction_col: str = 'prediction',
    lower_col: str = 'lower_bound',
    upper_col: str = 'upper_bound',
    top_n: int = 50
) -> plt.Figure:
    """
    绘制预测区间图（回归）

    Args:
        results_df: 结果DataFrame
        prediction_col: 预测列名
        lower_col: 下界列名
        upper_col: 上界列名
        top_n: 显示前N个样本

    Returns:
        fig: matplotlib Figure对象
    """
    # 选择前N个
    top_df = results_df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    x_pos = np.arange(len(top_df))

    # 绘制预测值
    ax.plot(x_pos, top_df[prediction_col], 'o-', color='blue', linewidth=2,
            markersize=6, label='Prediction')

    # 绘制置信区间
    if lower_col in top_df.columns and upper_col in top_df.columns:
        ax.fill_between(x_pos, top_df[lower_col], top_df[upper_col],
                        alpha=0.3, color='blue', label='95% Confidence Interval')

    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)
    ax.set_title(f'Prediction with Confidence Intervals (Top {top_n})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_prediction_vs_confidence(
    results_df: pd.DataFrame,
    prediction_col: str = 'prediction',
    confidence_col: str = 'confidence'
) -> plt.Figure:
    """
    绘制预测值vs置信度散点图

    Args:
        results_df: 结果DataFrame
        prediction_col: 预测列名
        confidence_col: 置信度列名

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # 散点图，颜色根据置信度
    scatter = ax.scatter(results_df[prediction_col], results_df[confidence_col],
                        c=results_df[confidence_col], cmap='RdYlGn',
                        s=50, alpha=0.6, edgecolors='k', linewidths=0.5)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Confidence Score', fontsize=12)

    # 添加置信度阈值线
    ax.axhline(y=0.7, color='green', linestyle='--', lw=2, alpha=0.5, label='High Confidence')
    ax.axhline(y=0.5, color='orange', linestyle='--', lw=2, alpha=0.5, label='Medium Confidence')

    ax.set_xlabel('Predicted Value', fontsize=12)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title('Prediction vs Confidence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

