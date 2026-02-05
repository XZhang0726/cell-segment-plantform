"""
主动学习模块

实现多种主动学习策略和贝叶斯优化，用于智能样本选择和模型优化
支持不确定性采样、委员会查询和贝叶斯优化等方法

主要功能：
1. 不确定性采样策略（最小置信度、边界、熵）
2. 委员会查询方法
3. 贝叶斯优化循环
4. 主动学习工作流
5. 精美的不确定性可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
from loguru import logger

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Scikit-optimize imports for Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

# Scipy imports
from scipy.stats import entropy

# ============================================================================
# UNCERTAINTY SAMPLING STRATEGIES
# ============================================================================

def uncertainty_sampling(
    model: Any,
    X_pool: np.ndarray,
    n_samples: int = 10,
    method: str = 'least_confident'
) -> np.ndarray:
    """
    基于不确定性的采样策略

    选择模型预测最不确定的样本进行标注

    Args:
        model: 训练好的分类模型（需要支持predict_proba）
        X_pool: 未标注样本池
        n_samples: 要选择的样本数量
        method: 不确定性度量方法
            - 'least_confident': 最小置信度（1 - max(p)）
            - 'margin': 边界采样（最大概率 - 第二大概率）
            - 'entropy': 熵采样（信息熵）

    Returns:
        selected_indices: 选中样本的索引数组
    """
    if not hasattr(model, 'predict_proba'):
        logger.error("Model does not support predict_proba, cannot use uncertainty sampling")
        raise ValueError("Model must support predict_proba for uncertainty sampling")

    # 获取预测概率
    probabilities = model.predict_proba(X_pool)

    # 计算不确定性分数
    if method == 'least_confident':
        # 最小置信度：1 - max(p)
        uncertainty_scores = 1 - np.max(probabilities, axis=1)

    elif method == 'margin':
        # 边界采样：最大概率 - 第二大概率（越小越不确定）
        if probabilities.shape[1] < 2:
            logger.warning("Only one class, using least_confident method instead")
            uncertainty_scores = 1 - np.max(probabilities, axis=1)
        else:
            # 对每行排序，取最大和第二大
            sorted_probs = np.sort(probabilities, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            uncertainty_scores = -margin  # 负号使得margin越小，分数越高

    elif method == 'entropy':
        # 熵采样：信息熵越大越不确定
        uncertainty_scores = entropy(probabilities.T)

    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

    # 选择不确定性最高的样本
    selected_indices = np.argsort(uncertainty_scores)[-n_samples:][::-1]

    logger.info(f"Selected {len(selected_indices)} samples using {method} uncertainty sampling")
    logger.info(f"Uncertainty scores range: [{uncertainty_scores.min():.4f}, {uncertainty_scores.max():.4f}]")

    return selected_indices


def query_by_committee(
    models: List[Any],
    X_pool: np.ndarray,
    n_samples: int = 10,
    disagreement: str = 'vote_entropy'
) -> np.ndarray:
    """
    委员会查询方法

    训练多个模型组成委员会，选择委员会分歧最大的样本

    Args:
        models: 模型列表（委员会成员）
        X_pool: 未标注样本池
        n_samples: 要选择的样本数量
        disagreement: 分歧度量方法
            - 'vote_entropy': 投票熵（分类）
            - 'variance': 预测方差（回归）

    Returns:
        selected_indices: 选中样本的索引数组
    """
    if len(models) < 2:
        logger.error("Need at least 2 models for query by committee")
        raise ValueError("Need at least 2 models for committee")

    # 检查是否为分类任务
    is_classification = hasattr(models[0], 'predict_proba')

    if disagreement == 'vote_entropy' and is_classification:
        # 获取所有模型的预测
        all_predictions = np.array([model.predict(X_pool) for model in models])

        # 计算投票熵
        disagreement_scores = []
        for i in range(X_pool.shape[0]):
            votes = all_predictions[:, i]
            # 计算每个类别的投票比例
            unique, counts = np.unique(votes, return_counts=True)
            vote_probs = counts / len(models)
            # 计算熵
            vote_entropy = entropy(vote_probs)
            disagreement_scores.append(vote_entropy)

        disagreement_scores = np.array(disagreement_scores)

    elif disagreement == 'variance':
        # 获取所有模型的预测
        all_predictions = np.array([model.predict(X_pool) for model in models])

        # 计算预测方差
        disagreement_scores = np.var(all_predictions, axis=0)

    else:
        raise ValueError(f"Unknown disagreement method: {disagreement}")

    # 选择分歧最大的样本
    selected_indices = np.argsort(disagreement_scores)[-n_samples:][::-1]

    logger.info(f"Selected {len(selected_indices)} samples using query by committee ({disagreement})")
    logger.info(f"Disagreement scores range: [{disagreement_scores.min():.4f}, {disagreement_scores.max():.4f}]")

    return selected_indices


def expected_improvement_sampling(
    model: Any,
    X_pool: np.ndarray,
    y_pool_estimated: np.ndarray,
    n_samples: int = 10
) -> np.ndarray:
    """
    期望改进采样

    选择预期能带来最大性能提升的样本

    Args:
        model: 训练好的模型
        X_pool: 未标注样本池
        y_pool_estimated: 样本池的估计目标值（用于计算期望改进）
        n_samples: 要选择的样本数量

    Returns:
        selected_indices: 选中样本的索引数组
    """
    # 获取预测
    predictions = model.predict(X_pool)

    # 计算期望改进（简化版本：基于预测值与估计值的差异）
    if hasattr(model, 'predict_proba'):
        # 分类任务：使用不确定性作为期望改进的代理
        probabilities = model.predict_proba(X_pool)
        uncertainty = 1 - np.max(probabilities, axis=1)
        expected_improvement = uncertainty
    else:
        # 回归任务：使用预测值与估计值的绝对差异
        expected_improvement = np.abs(predictions - y_pool_estimated)

    # 选择期望改进最大的样本
    selected_indices = np.argsort(expected_improvement)[-n_samples:][::-1]

    logger.info(f"Selected {len(selected_indices)} samples using expected improvement sampling")
    logger.info(f"Expected improvement range: [{expected_improvement.min():.4f}, {expected_improvement.max():.4f}]")

    return selected_indices


# ============================================================================
# BAYESIAN OPTIMIZATION FUNCTIONS
# ============================================================================

def fit_gaussian_process(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: Optional[Any] = None
) -> Any:
    """
    拟合高斯过程用于不确定性估计

    Args:
        X_train: 训练特征
        y_train: 训练标签
        kernel: 核函数（如果为None，使用默认RBF核）

    Returns:
        gp_model: 拟合好的高斯过程模型
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    if kernel is None:
        # 默认核：常数核 * RBF核
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

    gp_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-6,
        normalize_y=True
    )

    gp_model.fit(X_train, y_train)

    logger.info(f"Gaussian Process fitted with kernel: {gp_model.kernel_}")

    return gp_model


def compute_acquisition_function(
    model: Any,
    X_pool: np.ndarray,
    acquisition: str = 'ei',
    xi: float = 0.01,
    kappa: float = 1.96
) -> np.ndarray:
    """
    计算采集函数值

    Args:
        model: 高斯过程模型或支持预测不确定性的模型
        X_pool: 候选样本池
        acquisition: 采集函数类型
            - 'ei': Expected Improvement (期望改进)
            - 'ucb': Upper Confidence Bound (上置信界)
            - 'pi': Probability of Improvement (改进概率)
        xi: EI和PI的探索参数
        kappa: UCB的探索参数

    Returns:
        acquisition_values: 采集函数值数组
    """
    from scipy.stats import norm

    # 获取预测均值和标准差
    if hasattr(model, 'predict') and hasattr(model, 'predict'):
        # 高斯过程模型
        try:
            mu, sigma = model.predict(X_pool, return_std=True)
        except:
            # 如果模型不支持return_std，使用预测值作为均值，标准差设为0
            mu = model.predict(X_pool)
            sigma = np.zeros_like(mu)
    else:
        raise ValueError("Model must support predict with return_std=True")

    # 避免除零
    sigma = np.maximum(sigma, 1e-9)

    if acquisition == 'ei':
        # Expected Improvement
        # 找到当前最优值
        if hasattr(model, 'y_train_'):
            f_best = np.max(model.y_train_)
        else:
            f_best = np.max(mu)

        # 计算改进
        improvement = mu - f_best - xi
        Z = improvement / sigma

        # 计算期望改进
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        acquisition_values = ei

    elif acquisition == 'ucb':
        # Upper Confidence Bound
        acquisition_values = mu + kappa * sigma

    elif acquisition == 'pi':
        # Probability of Improvement
        if hasattr(model, 'y_train_'):
            f_best = np.max(model.y_train_)
        else:
            f_best = np.max(mu)

        improvement = mu - f_best - xi
        Z = improvement / sigma
        acquisition_values = norm.cdf(Z)

    else:
        raise ValueError(f"Unknown acquisition function: {acquisition}")

    logger.info(f"Computed {acquisition} acquisition function: range [{acquisition_values.min():.4f}, {acquisition_values.max():.4f}]")

    return acquisition_values


def bayesian_optimization_loop(
    objective_function: Optional[Callable] = None,
    X_train_initial: np.ndarray = None,
    y_train_initial: np.ndarray = None,
    X_pool: np.ndarray = None,
    n_iterations: int = 10,
    acquisition: str = 'ei',
    model_type: str = 'gp',
    samples_per_iteration: int = 1,
    random_state: int = 42
) -> Dict:
    """
    贝叶斯优化主循环

    Args:
        objective_function: 目标函数（如果为None，使用监督学习模式）
        X_train_initial: 初始训练特征
        y_train_initial: 初始训练标签
        X_pool: 候选样本池
        n_iterations: 优化迭代次数
        acquisition: 采集函数 ('ei', 'ucb', 'pi')
        model_type: 代理模型类型 ('gp', 'rf', 'gbrt')
        samples_per_iteration: 每次迭代选择的样本数
        random_state: 随机种子

    Returns:
        results: 包含优化历史的字典
            - 'selected_samples': 每次迭代选中的样本索引
            - 'acquisition_values': 每次迭代的采集函数值
            - 'best_values': 每次迭代的最优值
            - 'model_history': 模型历史
            - 'X_train_history': 训练集历史
            - 'y_train_history': 标签历史
    """
    np.random.seed(random_state)

    # 初始化
    X_train = X_train_initial.copy()
    y_train = y_train_initial.copy()
    X_pool_remaining = X_pool.copy()

    # 记录历史
    selected_samples_history = []
    acquisition_values_history = []
    best_values_history = []
    model_history = []
    X_train_history = [X_train.copy()]
    y_train_history = [y_train.copy()]

    logger.info(f"Starting Bayesian optimization: {n_iterations} iterations, acquisition={acquisition}")

    for iteration in range(n_iterations):
        logger.info(f"Iteration {iteration + 1}/{n_iterations}")

        # 训练代理模型
        if model_type == 'gp':
            model = fit_gaussian_process(X_train, y_train)
        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            model.fit(X_train, y_train)
        elif model_type == 'gbrt':
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
            model.fit(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 计算采集函数
        if model_type == 'gp':
            acq_values = compute_acquisition_function(model, X_pool_remaining, acquisition=acquisition)
        else:
            # 对于非GP模型，使用简单的不确定性估计
            predictions = model.predict(X_pool_remaining)
            acq_values = predictions  # 简化版本

        # 选择样本
        selected_indices = np.argsort(acq_values)[-samples_per_iteration:][::-1]
        selected_samples_history.append(selected_indices)
        acquisition_values_history.append(acq_values)

        # 获取选中样本的真实标签
        X_selected = X_pool_remaining[selected_indices]

        if objective_function is not None:
            # 使用目标函数评估
            y_selected = np.array([objective_function(x) for x in X_selected])
        else:
            # 监督学习模式：需要从外部提供标签
            logger.warning("No objective function provided, cannot evaluate selected samples")
            y_selected = np.zeros(len(selected_indices))

        # 更新训练集
        X_train = np.vstack([X_train, X_selected])
        y_train = np.concatenate([y_train, y_selected])

        # 从池中移除选中的样本
        mask = np.ones(len(X_pool_remaining), dtype=bool)
        mask[selected_indices] = False
        X_pool_remaining = X_pool_remaining[mask]

        # 记录最优值
        best_value = np.max(y_train)
        best_values_history.append(best_value)

        # 记录历史
        model_history.append(model)
        X_train_history.append(X_train.copy())
        y_train_history.append(y_train.copy())

        logger.info(f"Selected {len(selected_indices)} samples, best value so far: {best_value:.4f}")

        # 如果池已空，提前结束
        if len(X_pool_remaining) == 0:
            logger.info("Pool exhausted, stopping optimization")
            break

    results = {
        'selected_samples': selected_samples_history,
        'acquisition_values': acquisition_values_history,
        'best_values': best_values_history,
        'model_history': model_history,
        'X_train_history': X_train_history,
        'y_train_history': y_train_history,
        'final_model': model_history[-1] if model_history else None,
        'n_iterations': len(best_values_history)
    }

    logger.info(f"Bayesian optimization completed: {len(best_values_history)} iterations")

    return results


# ============================================================================
# ACTIVE LEARNING WORKFLOW
# ============================================================================

def active_learning_workflow(
    X_train_initial: np.ndarray,
    y_train_initial: np.ndarray,
    X_pool: np.ndarray,
    y_pool_true: np.ndarray,
    model_name: str = 'random_forest',
    task_type: str = 'classification',
    strategy: str = 'uncertainty',
    n_iterations: int = 10,
    samples_per_iteration: int = 10,
    random_state: int = 42
) -> Dict:
    """
    完整的主动学习工作流

    Args:
        X_train_initial: 初始训练特征
        y_train_initial: 初始训练标签
        X_pool: 未标注样本池
        y_pool_true: 样本池的真实标签（用于模拟）
        model_name: 模型名称 ('random_forest', 'svm', 'logistic', etc.)
        task_type: 任务类型 ('classification', 'regression')
        strategy: 采样策略 ('uncertainty', 'qbc', 'random')
        n_iterations: 迭代次数
        samples_per_iteration: 每次迭代选择的样本数
        random_state: 随机种子

    Returns:
        results: 包含完整历史的字典
            - 'iteration_metrics': 每次迭代的性能指标
            - 'selected_indices': 每次迭代选中的样本索引
            - 'final_model': 最终训练的模型
            - 'training_history': 完整训练历史
            - 'X_train_history': 训练集历史
            - 'y_train_history': 标签历史
    """
    np.random.seed(random_state)

    # 初始化
    X_train = X_train_initial.copy()
    y_train = y_train_initial.copy()
    X_pool_remaining = X_pool.copy()
    y_pool_remaining = y_pool_true.copy()

    # 记录历史
    iteration_metrics = []
    selected_indices_history = []
    X_train_history = [X_train.copy()]
    y_train_history = [y_train.copy()]

    # 选择模型
    if task_type == 'classification':
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_name == 'svm':
            from sklearn.svm import SVC
            base_model = SVC(probability=True, random_state=random_state)
        elif model_name == 'logistic':
            from sklearn.linear_model import LogisticRegression
            base_model = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            base_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif model_name == 'svm':
            from sklearn.svm import SVR
            base_model = SVR()
        else:
            from sklearn.ensemble import RandomForestRegressor
            base_model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    logger.info(f"Starting active learning: {n_iterations} iterations, strategy={strategy}")

    for iteration in range(n_iterations):
        logger.info(f"Iteration {iteration + 1}/{n_iterations}")

        # 训练模型
        model = base_model.__class__(**base_model.get_params())
        model.fit(X_train, y_train)

        # 评估当前模型
        if task_type == 'classification':
            train_score = accuracy_score(y_train, model.predict(X_train))
            # 在整个池上评估（包括已标注和未标注）
            all_X = np.vstack([X_train, X_pool_remaining])
            all_y = np.concatenate([y_train, y_pool_remaining])
            test_score = accuracy_score(all_y, model.predict(all_X))
            metric_name = 'accuracy'
        else:
            train_score = r2_score(y_train, model.predict(X_train))
            all_X = np.vstack([X_train, X_pool_remaining])
            all_y = np.concatenate([y_train, y_pool_remaining])
            test_score = r2_score(all_y, model.predict(all_X))
            metric_name = 'r2_score'

        iteration_metrics.append({
            'iteration': iteration + 1,
            'n_train': len(X_train),
            'train_score': train_score,
            'test_score': test_score,
            'metric_name': metric_name
        })

        logger.info(f"Train {metric_name}: {train_score:.4f}, Test {metric_name}: {test_score:.4f}")

        # 如果池已空，提前结束
        if len(X_pool_remaining) == 0:
            logger.info("Pool exhausted, stopping active learning")
            break

        # 选择样本
        if strategy == 'uncertainty':
            if task_type == 'classification':
                selected_indices = uncertainty_sampling(
                    model, X_pool_remaining,
                    n_samples=min(samples_per_iteration, len(X_pool_remaining)),
                    method='entropy'
                )
            else:
                # 回归任务：随机采样（简化版本）
                selected_indices = np.random.choice(
                    len(X_pool_remaining),
                    size=min(samples_per_iteration, len(X_pool_remaining)),
                    replace=False
                )

        elif strategy == 'qbc':
            # 训练委员会
            committee = []
            for i in range(3):
                committee_model = base_model.__class__(**base_model.get_params())
                # 使用bootstrap采样
                bootstrap_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                committee_model.fit(X_train[bootstrap_indices], y_train[bootstrap_indices])
                committee.append(committee_model)

            selected_indices = query_by_committee(
                committee, X_pool_remaining,
                n_samples=min(samples_per_iteration, len(X_pool_remaining)),
                disagreement='vote_entropy' if task_type == 'classification' else 'variance'
            )

        elif strategy == 'random':
            # 随机采样（基线）
            selected_indices = np.random.choice(
                len(X_pool_remaining),
                size=min(samples_per_iteration, len(X_pool_remaining)),
                replace=False
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        selected_indices_history.append(selected_indices)

        # 获取选中样本的真实标签
        X_selected = X_pool_remaining[selected_indices]
        y_selected = y_pool_remaining[selected_indices]

        # 更新训练集
        X_train = np.vstack([X_train, X_selected])
        y_train = np.concatenate([y_train, y_selected])

        # 从池中移除选中的样本
        mask = np.ones(len(X_pool_remaining), dtype=bool)
        mask[selected_indices] = False
        X_pool_remaining = X_pool_remaining[mask]
        y_pool_remaining = y_pool_remaining[mask]

        # 记录历史
        X_train_history.append(X_train.copy())
        y_train_history.append(y_train.copy())

    # 训练最终模型
    final_model = base_model.__class__(**base_model.get_params())
    final_model.fit(X_train, y_train)

    results = {
        'iteration_metrics': iteration_metrics,
        'selected_indices': selected_indices_history,
        'final_model': final_model,
        'training_history': {
            'X_train': X_train_history,
            'y_train': y_train_history
        },
        'X_train_history': X_train_history,
        'y_train_history': y_train_history,
        'n_iterations': len(iteration_metrics),
        'strategy': strategy,
        'task_type': task_type
    }

    logger.info(f"Active learning completed: {len(iteration_metrics)} iterations")

    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_uncertainty_intervals(
    X: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    X_new: Optional[np.ndarray] = None,
    y_new: Optional[np.ndarray] = None,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    title: str = 'Uncertainty Intervals',
    feature_idx: int = 0
) -> plt.Figure:
    """
    绘制精美的不确定性区间图

    使用fill_between创建阴影置信区间，展示模型预测的不确定性

    Args:
        X: 预测点的特征
        y_pred: 预测均值
        y_std: 预测标准差
        X_new: 新选择的样本特征（可选）
        y_new: 新选择的样本标签（可选）
        X_train: 训练样本特征（可选）
        y_train: 训练样本标签（可选）
        title: 图表标题
        feature_idx: 用于绘图的特征索引（如果X是多维的）

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # 如果X是多维的，只使用指定的特征维度
    if X.ndim > 1:
        X_plot = X[:, feature_idx]
    else:
        X_plot = X

    # 排序以便平滑绘图
    sort_idx = np.argsort(X_plot)
    X_sorted = X_plot[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    y_std_sorted = y_std[sort_idx]

    # 绘制预测均值
    ax.plot(X_sorted, y_pred_sorted, 'b-', linewidth=2.5, label='Mean Prediction', zorder=3)

    # 绘制95%置信区间（1.96 * std）
    ax.fill_between(
        X_sorted,
        y_pred_sorted - 1.96 * y_std_sorted,
        y_pred_sorted + 1.96 * y_std_sorted,
        alpha=0.2, color='blue', label='95% Confidence', zorder=1
    )

    # 绘制68%置信区间（1 * std）
    ax.fill_between(
        X_sorted,
        y_pred_sorted - y_std_sorted,
        y_pred_sorted + y_std_sorted,
        alpha=0.3, color='blue', label='68% Confidence', zorder=2
    )

    # 绘制训练样本
    if X_train is not None and y_train is not None:
        if X_train.ndim > 1:
            X_train_plot = X_train[:, feature_idx]
        else:
            X_train_plot = X_train
        ax.scatter(X_train_plot, y_train, c='green', s=80, marker='o',
                   edgecolors='black', linewidths=1, alpha=0.7,
                   label='Training Samples', zorder=4)

    # 绘制新选择的样本
    if X_new is not None and y_new is not None:
        if X_new.ndim > 1:
            X_new_plot = X_new[:, feature_idx]
        else:
            X_new_plot = X_new
        ax.scatter(X_new_plot, y_new, c='red', s=150, marker='*',
                   edgecolors='black', linewidths=1.5,
                   label='Selected Samples', zorder=5)

    ax.set_xlabel('Feature Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Target Value', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_acquisition_function(
    X_pool: np.ndarray,
    acquisition_values: np.ndarray,
    selected_idx: Optional[np.ndarray] = None,
    title: str = 'Acquisition Function',
    feature_idx: int = 0
) -> plt.Figure:
    """
    绘制采集函数图

    Args:
        X_pool: 候选样本池
        acquisition_values: 采集函数值
        selected_idx: 选中样本的索引（可选）
        title: 图表标题
        feature_idx: 用于绘图的特征索引

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # 如果X是多维的，只使用指定的特征维度
    if X_pool.ndim > 1:
        X_plot = X_pool[:, feature_idx]
    else:
        X_plot = X_pool

    # 排序以便平滑绘图
    sort_idx = np.argsort(X_plot)
    X_sorted = X_plot[sort_idx]
    acq_sorted = acquisition_values[sort_idx]

    # 绘制采集函数
    ax.plot(X_sorted, acq_sorted, 'g-', linewidth=2, label='Acquisition Function')
    ax.fill_between(X_sorted, 0, acq_sorted, alpha=0.3, color='green')

    # 高亮选中的样本
    if selected_idx is not None:
        X_selected = X_plot[selected_idx]
        acq_selected = acquisition_values[selected_idx]
        ax.scatter(X_selected, acq_selected, c='red', s=150, marker='*',
                   edgecolors='black', linewidths=1.5,
                   label='Selected Samples', zorder=5)

    ax.set_xlabel('Feature Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Acquisition Value', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_optimization_trajectory(
    results: Dict,
    metric: str = 'test_score'
) -> plt.Figure:
    """
    绘制优化轨迹图

    展示模型性能随迭代次数的变化

    Args:
        results: active_learning_workflow返回的结果字典
        metric: 要绘制的指标 ('test_score', 'train_score')

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    iteration_metrics = results['iteration_metrics']
    iterations = [m['iteration'] for m in iteration_metrics]

    if metric == 'test_score':
        scores = [m['test_score'] for m in iteration_metrics]
        label = 'Test Score'
    elif metric == 'train_score':
        scores = [m['train_score'] for m in iteration_metrics]
        label = 'Train Score'
    else:
        scores = [m.get(metric, 0) for m in iteration_metrics]
        label = metric

    # 绘制性能曲线
    ax.plot(iterations, scores, 'o-', linewidth=2.5, markersize=8,
            color='steelblue', label=label)

    # 添加最佳性能线
    best_score = max(scores)
    best_iter = iterations[scores.index(best_score)]
    ax.axhline(y=best_score, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Best: {best_score:.4f} (Iter {best_iter})')

    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax.set_title('Optimization Trajectory', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_convergence(
    results: Dict,
    show_confidence: bool = True
) -> plt.Figure:
    """
    绘制收敛图

    展示性能改进和收敛趋势

    Args:
        results: active_learning_workflow或bayesian_optimization_loop返回的结果字典
        show_confidence: 是否显示置信带

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 检查结果类型
    if 'iteration_metrics' in results:
        # Active learning结果
        iteration_metrics = results['iteration_metrics']
        iterations = [m['iteration'] for m in iteration_metrics]
        scores = [m['test_score'] for m in iteration_metrics]
        ylabel = 'Test Score'
    elif 'best_values' in results:
        # Bayesian optimization结果
        iterations = list(range(1, len(results['best_values']) + 1))
        scores = results['best_values']
        ylabel = 'Best Value'
    else:
        raise ValueError("Unknown results format")

    # 绘制收敛曲线
    ax.plot(iterations, scores, 'o-', linewidth=2.5, markersize=8,
            color='darkgreen', label='Performance')

    # 如果显示置信带，计算移动平均和标准差
    if show_confidence and len(scores) > 3:
        window = min(3, len(scores))
        moving_avg = pd.Series(scores).rolling(window=window, center=True, min_periods=1).mean()
        moving_std = pd.Series(scores).rolling(window=window, center=True, min_periods=1).std()

        ax.plot(iterations, moving_avg, '--', linewidth=2, color='orange',
                label=f'Moving Average (window={window})')

        if not moving_std.isna().all():
            ax.fill_between(iterations,
                           moving_avg - moving_std,
                           moving_avg + moving_std,
                           alpha=0.2, color='orange')

    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title('Convergence Plot', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_learning_progress(
    iteration_metrics: List[Dict],
    metrics_to_plot: List[str] = ['test_score', 'train_score']
) -> plt.Figure:
    """
    绘制学习进度图

    展示多个指标随迭代的变化

    Args:
        iteration_metrics: 迭代指标列表
        metrics_to_plot: 要绘制的指标列表

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    iterations = [m['iteration'] for m in iteration_metrics]

    colors = ['steelblue', 'darkgreen', 'coral', 'purple']

    for i, metric in enumerate(metrics_to_plot):
        if metric in iteration_metrics[0]:
            values = [m[metric] for m in iteration_metrics]
            color = colors[i % len(colors)]
            ax.plot(iterations, values, 'o-', linewidth=2, markersize=6,
                   color=color, label=metric.replace('_', ' ').title())

    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Learning Progress', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_exploration_space_2d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    selected_idx: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    feature_indices: Tuple[int, int] = (0, 1)
) -> plt.Figure:
    """
    绘制2D探索空间可视化

    Args:
        X_train: 训练样本特征
        y_train: 训练样本标签
        X_pool: 样本池特征
        selected_idx: 选中样本的索引（可选）
        feature_names: 特征名称列表（可选）
        feature_indices: 用于绘图的两个特征索引

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    idx1, idx2 = feature_indices

    # 绘制训练样本
    scatter_train = ax.scatter(
        X_train[:, idx1], X_train[:, idx2],
        c=y_train, cmap='viridis', s=100,
        edgecolors='black', linewidths=1.5,
        alpha=0.8, label='Training Samples'
    )

    # 绘制样本池
    ax.scatter(
        X_pool[:, idx1], X_pool[:, idx2],
        c='lightgray', s=50, alpha=0.5,
        edgecolors='gray', linewidths=0.5,
        label='Pool Samples'
    )

    # 绘制选中的样本
    if selected_idx is not None:
        ax.scatter(
            X_pool[selected_idx, idx1], X_pool[selected_idx, idx2],
            c='red', s=200, marker='*',
            edgecolors='black', linewidths=2,
            label='Selected Samples', zorder=5
        )

    # 设置标签
    if feature_names is not None:
        xlabel = feature_names[idx1]
        ylabel = feature_names[idx2]
    else:
        xlabel = f'Feature {idx1}'
        ylabel = f'Feature {idx2}'

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title('2D Exploration Space', fontsize=16, fontweight='bold', pad=20)

    # 添加颜色条
    cbar = plt.colorbar(scatter_train, ax=ax)
    cbar.set_label('Target Value', fontsize=12)

    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_exploration_space_3d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    selected_idx: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    feature_indices: Tuple[int, int, int] = (0, 1, 2)
) -> plt.Figure:
    """
    绘制3D探索空间可视化

    Args:
        X_train: 训练样本特征
        y_train: 训练样本标签
        X_pool: 样本池特征
        selected_idx: 选中样本的索引（可选）
        feature_names: 特征名称列表（可选）
        feature_indices: 用于绘图的三个特征索引

    Returns:
        fig: matplotlib Figure对象
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    idx1, idx2, idx3 = feature_indices

    # 绘制训练样本
    scatter_train = ax.scatter(
        X_train[:, idx1], X_train[:, idx2], X_train[:, idx3],
        c=y_train, cmap='viridis', s=100,
        edgecolors='black', linewidths=1.5,
        alpha=0.8, label='Training Samples'
    )

    # 绘制样本池
    ax.scatter(
        X_pool[:, idx1], X_pool[:, idx2], X_pool[:, idx3],
        c='lightgray', s=30, alpha=0.3,
        edgecolors='gray', linewidths=0.5,
        label='Pool Samples'
    )

    # 绘制选中的样本
    if selected_idx is not None:
        ax.scatter(
            X_pool[selected_idx, idx1],
            X_pool[selected_idx, idx2],
            X_pool[selected_idx, idx3],
            c='red', s=200, marker='*',
            edgecolors='black', linewidths=2,
            label='Selected Samples'
        )

    # 设置标签
    if feature_names is not None:
        xlabel = feature_names[idx1]
        ylabel = feature_names[idx2]
        zlabel = feature_names[idx3]
    else:
        xlabel = f'Feature {idx1}'
        ylabel = f'Feature {idx2}'
        zlabel = f'Feature {idx3}'

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_zlabel(zlabel, fontsize=12, fontweight='bold')
    ax.set_title('3D Exploration Space', fontsize=16, fontweight='bold', pad=20)

    # 添加颜色条
    cbar = plt.colorbar(scatter_train, ax=ax, shrink=0.8)
    cbar.set_label('Target Value', fontsize=12)

    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    return fig


def plot_sample_selection_heatmap(
    selected_indices_history: List[np.ndarray],
    n_pool_samples: int
) -> plt.Figure:
    """
    绘制样本选择热图

    展示哪些样本在哪些迭代中被选中

    Args:
        selected_indices_history: 每次迭代选中的样本索引列表
        n_pool_samples: 样本池总数

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # 创建选择矩阵
    n_iterations = len(selected_indices_history)
    selection_matrix = np.zeros((n_iterations, n_pool_samples))

    for i, selected_idx in enumerate(selected_indices_history):
        selection_matrix[i, selected_idx] = 1

    # 绘制热图
    im = ax.imshow(selection_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    # 设置标签
    ax.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_title('Sample Selection Heatmap', fontsize=16, fontweight='bold', pad=20)

    # 设置刻度
    ax.set_yticks(range(n_iterations))
    ax.set_yticklabels([f'Iter {i+1}' for i in range(n_iterations)])

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selected', fontsize=12)

    plt.tight_layout()
    return fig
