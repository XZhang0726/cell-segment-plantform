"""
监督学习模块

提供完整的监督学习流程，包括特征工程、模型训练、评估和预测
支持分类和回归任务，提供AutoML功能自动选择最佳模型

主要功能：
1. 特征工程：特征选择、缩放、编码、多项式特征生成
2. 模型训练：支持多种sklearn模型，超参数调优
3. 模型评估：全面的评估指标和可视化
4. 模型持久化：保存/加载模型及元数据
5. AutoML：自动模型比较和选择
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import joblib
from pathlib import Path
import warnings

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, RFE, f_classif, f_regression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)

# Classification models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Regression models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

from loguru import logger

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# CONSTANTS AND CONFIGURATIONS
# ============================================================================

# Classification models registry
CLASSIFICATION_MODELS = {
    'random_forest': RandomForestClassifier,
    'svm': SVC,
    'logistic': LogisticRegression,
    'gradient_boosting': GradientBoostingClassifier,
    'knn': KNeighborsClassifier,
    'decision_tree': DecisionTreeClassifier,
    'adaboost': AdaBoostClassifier,
    'extra_trees': ExtraTreesClassifier,
}

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS['xgboost'] = XGBClassifier

# Regression models registry
REGRESSION_MODELS = {
    'random_forest': RandomForestRegressor,
    'svr': SVR,
    'linear': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso,
    'elastic_net': ElasticNet,
    'gradient_boosting': GradientBoostingRegressor,
    'decision_tree': DecisionTreeRegressor,
}

if XGBOOST_AVAILABLE:
    REGRESSION_MODELS['xgboost'] = XGBRegressor

# Default hyperparameter grids for classification
PARAM_GRIDS_CLASSIFICATION = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    },
    'logistic': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'decision_tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    'adaboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'extra_trees': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
}

if XGBOOST_AVAILABLE:
    PARAM_GRIDS_CLASSIFICATION['xgboost'] = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

# Default hyperparameter grids for regression
PARAM_GRIDS_REGRESSION = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'svr': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'epsilon': [0.01, 0.1, 0.2]
    },
    'ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    },
    'lasso': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    },
    'elastic_net': {
        'alpha': [0.01, 0.1, 1, 10],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    },
    'decision_tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
}

if XGBOOST_AVAILABLE:
    PARAM_GRIDS_REGRESSION['xgboost'] = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

# Default model parameters
DEFAULT_PARAMS_CLASSIFICATION = {
    'random_forest': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
    'svm': {'probability': True, 'random_state': 42},
    'logistic': {'random_state': 42, 'max_iter': 1000},
    'gradient_boosting': {'n_estimators': 100, 'random_state': 42},
    'knn': {'n_neighbors': 5},
    'decision_tree': {'random_state': 42},
    'adaboost': {'n_estimators': 100, 'random_state': 42},
    'extra_trees': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
}

if XGBOOST_AVAILABLE:
    DEFAULT_PARAMS_CLASSIFICATION['xgboost'] = {
        'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'
    }

DEFAULT_PARAMS_REGRESSION = {
    'random_forest': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
    'svr': {},
    'linear': {},
    'ridge': {'random_state': 42},
    'lasso': {'random_state': 42},
    'elastic_net': {'random_state': 42},
    'gradient_boosting': {'n_estimators': 100, 'random_state': 42},
    'decision_tree': {'random_state': 42},
}

if XGBOOST_AVAILABLE:
    DEFAULT_PARAMS_REGRESSION['xgboost'] = {
        'n_estimators': 100, 'random_state': 42, 'n_jobs': -1
    }


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def select_features_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.1,
    method: str = 'pearson'
) -> Tuple[pd.DataFrame, Dict]:
    """
    基于相关性的特征选择

    Args:
        X: 特征DataFrame
        y: 目标变量Series
        threshold: 相关性阈值（绝对值）
        method: 相关性计算方法 ('pearson', 'spearman', 'kendall')

    Returns:
        X_selected: 选择后的特征DataFrame
        info: 包含特征相关性信息的字典
    """
    logger.info(f"Selecting features by correlation (method={method}, threshold={threshold})")

    # 计算特征与目标的相关性
    correlations = {}
    for col in X.columns:
        if method == 'pearson':
            corr = X[col].corr(y, method='pearson')
        elif method == 'spearman':
            corr = X[col].corr(y, method='spearman')
        elif method == 'kendall':
            corr = X[col].corr(y, method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        correlations[col] = corr

    # 选择相关性高于阈值的特征
    selected_features = [col for col, corr in correlations.items() if abs(corr) >= threshold]

    if len(selected_features) == 0:
        logger.warning(f"No features meet correlation threshold {threshold}, keeping all features")
        selected_features = list(X.columns)

    X_selected = X[selected_features]

    info = {
        'method': 'correlation',
        'correlation_method': method,
        'threshold': threshold,
        'n_features_original': len(X.columns),
        'n_features_selected': len(selected_features),
        'selected_features': selected_features,
        'correlations': correlations
    }

    logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")

    return X_selected, info


def select_features_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: Optional[int] = None,
    task_type: str = 'classification'
) -> Tuple[pd.DataFrame, Dict]:
    """
    基于互信息的特征选择

    Args:
        X: 特征DataFrame
        y: 目标变量Series
        n_features: 要选择的特征数量（None表示自动选择前50%）
        task_type: 'classification' or 'regression'

    Returns:
        X_selected: 选择后的特征DataFrame
        info: 包含特征重要性信息的字典
    """
    logger.info(f"Selecting features by mutual information (task_type={task_type})")

    if n_features is None:
        n_features = max(1, len(X.columns) // 2)

    n_features = min(n_features, len(X.columns))

    # 计算互信息
    if task_type == 'classification':
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)

    # 创建特征重要性DataFrame
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    # 选择前n个特征
    selected_features = mi_df.head(n_features)['feature'].tolist()
    X_selected = X[selected_features]

    info = {
        'method': 'mutual_info',
        'task_type': task_type,
        'n_features_original': len(X.columns),
        'n_features_selected': len(selected_features),
        'selected_features': selected_features,
        'mi_scores': dict(zip(mi_df['feature'], mi_df['mi_score']))
    }

    logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")

    return X_selected, info


def select_features_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: Optional[int] = None,
    estimator: Optional[Any] = None,
    task_type: str = 'classification'
) -> Tuple[pd.DataFrame, Dict]:
    """
    递归特征消除（RFE）

    Args:
        X: 特征DataFrame
        y: 目标变量Series
        n_features: 要选择的特征数量（None表示自动选择前50%）
        estimator: 用于特征选择的估计器（None使用默认）
        task_type: 'classification' or 'regression'

    Returns:
        X_selected: 选择后的特征DataFrame
        info: 包含特征排名信息的字典
    """
    logger.info(f"Selecting features by RFE (task_type={task_type})")

    if n_features is None:
        n_features = max(1, len(X.columns) // 2)

    n_features = min(n_features, len(X.columns))

    # 使用默认估计器
    if estimator is None:
        if task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

    # 执行RFE
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X, y)

    # 获取选择的特征
    selected_features = X.columns[rfe.support_].tolist()
    X_selected = X[selected_features]

    # 获取特征排名
    feature_ranking = dict(zip(X.columns, rfe.ranking_))

    info = {
        'method': 'rfe',
        'task_type': task_type,
        'n_features_original': len(X.columns),
        'n_features_selected': len(selected_features),
        'selected_features': selected_features,
        'feature_ranking': feature_ranking
    }

    logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")

    return X_selected, info


def select_features_tree_based(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: Optional[int] = None,
    threshold: str = 'median',
    task_type: str = 'classification'
) -> Tuple[pd.DataFrame, Dict]:
    """
    基于树模型的特征选择

    Args:
        X: 特征DataFrame
        y: 目标变量Series
        n_features: 要选择的特征数量（None表示使用threshold）
        threshold: 重要性阈值 ('mean', 'median', or float value)
        task_type: 'classification' or 'regression'

    Returns:
        X_selected: 选择后的特征DataFrame
        info: 包含特征重要性信息的字典
    """
    logger.info(f"Selecting features by tree-based importance (task_type={task_type})")

    # 训练树模型
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X, y)

    # 获取特征重要性
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # 选择特征
    if n_features is not None:
        n_features = min(n_features, len(X.columns))
        selected_features = importance_df.head(n_features)['feature'].tolist()
    else:
        # 使用阈值
        if threshold == 'mean':
            thresh_value = importances.mean()
        elif threshold == 'median':
            thresh_value = np.median(importances)
        else:
            thresh_value = float(threshold)

        selected_features = importance_df[importance_df['importance'] >= thresh_value]['feature'].tolist()

        if len(selected_features) == 0:
            logger.warning(f"No features meet threshold {thresh_value}, keeping top 50%")
            n_features = max(1, len(X.columns) // 2)
            selected_features = importance_df.head(n_features)['feature'].tolist()

    X_selected = X[selected_features]

    info = {
        'method': 'tree_based',
        'task_type': task_type,
        'threshold': threshold,
        'n_features_original': len(X.columns),
        'n_features_selected': len(selected_features),
        'selected_features': selected_features,
        'feature_importance': dict(zip(importance_df['feature'], importance_df['importance']))
    }

    logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")

    return X_selected, info


def scale_features(
    X: Union[pd.DataFrame, np.ndarray],
    method: str = 'standard',
    scaler: Optional[Any] = None,
    fit: bool = True
) -> Tuple[np.ndarray, Any]:
    """
    特征缩放

    Args:
        X: 特征DataFrame或数组
        method: 缩放方法 ('standard', 'minmax', 'robust')
        scaler: 已拟合的缩放器（None表示创建新的）
        fit: 是否拟合scaler（训练时True，预测时False）

    Returns:
        X_scaled: 缩放后的特征数组
        scaler: 缩放器对象
    """
    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    if fit:
        X_scaled = scaler.fit_transform(X)
        logger.info(f"Features scaled using {method} method (fitted)")
    else:
        X_scaled = scaler.transform(X)
        logger.info(f"Features scaled using {method} method (transform only)")

    return X_scaled, scaler


def encode_categorical(
    X: pd.DataFrame,
    method: str = 'onehot',
    encoder: Optional[Any] = None,
    fit: bool = True,
    categorical_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Any]:
    """
    分类变量编码

    Args:
        X: 特征DataFrame
        method: 编码方法 ('onehot', 'label')
        encoder: 已拟合的编码器（None表示创建新的）
        fit: 是否拟合encoder（训练时True，预测时False）
        categorical_cols: 分类列名列表（None表示自动检测）

    Returns:
        X_encoded: 编码后的DataFrame
        encoder: 编码器对象
    """
    # 自动检测分类列
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(categorical_cols) == 0:
        logger.info("No categorical columns found, returning original DataFrame")
        return X, None

    logger.info(f"Encoding {len(categorical_cols)} categorical columns using {method} method")

    if method == 'onehot':
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        if fit:
            encoded = encoder.fit_transform(X[categorical_cols])
        else:
            encoded = encoder.transform(X[categorical_cols])

        # 创建新的列名
        feature_names = encoder.get_feature_names_out(categorical_cols)

        # 创建编码后的DataFrame
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)

        # 合并非分类列
        non_categorical_cols = [col for col in X.columns if col not in categorical_cols]
        X_encoded = pd.concat([X[non_categorical_cols], encoded_df], axis=1)

    elif method == 'label':
        X_encoded = X.copy()
        if encoder is None:
            encoder = {}

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                encoder[col] = le
            else:
                le = encoder.get(col)
                if le is not None:
                    X_encoded[col] = le.transform(X[col].astype(str))

    else:
        raise ValueError(f"Unknown encoding method: {method}")

    logger.info(f"Categorical encoding completed: {X_encoded.shape[1]} features")

    return X_encoded, encoder


def generate_polynomial_features(
    X: pd.DataFrame,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    生成多项式特征

    Args:
        X: 特征DataFrame
        degree: 多项式次数
        interaction_only: 是否仅生成交互项
        include_bias: 是否包含偏置项

    Returns:
        X_poly: 包含多项式特征的DataFrame
        feature_names: 新特征名称列表
    """
    from sklearn.preprocessing import PolynomialFeatures

    logger.info(f"Generating polynomial features (degree={degree}, interaction_only={interaction_only})")

    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    X_poly_array = poly.fit_transform(X)

    # 生成特征名称
    feature_names = poly.get_feature_names_out(X.columns)

    X_poly = pd.DataFrame(X_poly_array, columns=feature_names, index=X.index)

    logger.info(f"Polynomial features generated: {X_poly.shape[1]} features (from {X.shape[1]} original)")

    return X_poly, feature_names.tolist()


def detect_feature_interactions(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 10,
    task_type: str = 'classification'
) -> pd.DataFrame:
    """
    检测特征交互

    使用随机森林检测特征之间的交互效应

    Args:
        X: 特征DataFrame
        y: 目标变量Series
        top_n: 返回前N个最重要的交互
        task_type: 'classification' or 'regression'

    Returns:
        interactions_df: 包含交互特征对和重要性的DataFrame
    """
    logger.info(f"Detecting feature interactions (top_n={top_n})")

    # 生成所有二阶交互特征
    from itertools import combinations

    interactions = []
    feature_pairs = list(combinations(X.columns, 2))

    # 限制交互数量以避免过多特征
    if len(feature_pairs) > 100:
        logger.warning(f"Too many feature pairs ({len(feature_pairs)}), sampling 100 pairs")
        import random
        random.seed(42)
        feature_pairs = random.sample(feature_pairs, 100)

    # 创建交互特征
    X_interactions = X.copy()
    for feat1, feat2 in feature_pairs:
        interaction_name = f"{feat1}_x_{feat2}"
        X_interactions[interaction_name] = X[feat1] * X[feat2]
        interactions.append((feat1, feat2, interaction_name))

    # 训练模型获取特征重要性
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_interactions, y)

    # 获取交互特征的重要性
    importances = model.feature_importances_
    feature_names = X_interactions.columns

    # 只保留交互特征的重要性
    interaction_importances = []
    for feat1, feat2, interaction_name in interactions:
        idx = list(feature_names).index(interaction_name)
        importance = importances[idx]
        interaction_importances.append({
            'feature_1': feat1,
            'feature_2': feat2,
            'interaction': interaction_name,
            'importance': importance
        })

    # 创建DataFrame并排序
    interactions_df = pd.DataFrame(interaction_importances)
    interactions_df = interactions_df.sort_values('importance', ascending=False).head(top_n)

    logger.info(f"Top {len(interactions_df)} feature interactions identified")

    return interactions_df


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    task_type: str = 'classification',
    params: Optional[Dict] = None,
    cv_folds: int = 5
) -> Tuple[Any, Dict]:
    """
    训练单个模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        model_name: 模型名称
        task_type: 'classification' or 'regression'
        params: 模型参数字典（None使用默认参数）
        cv_folds: 交叉验证折数

    Returns:
        model: 训练好的模型
        info: 包含训练信息、CV分数的字典
    """
    logger.info(f"Training {model_name} model (task_type={task_type})")

    # 获取模型类
    if task_type == 'classification':
        if model_name not in CLASSIFICATION_MODELS:
            raise ValueError(f"Unknown classification model: {model_name}")
        model_class = CLASSIFICATION_MODELS[model_name]
        default_params = DEFAULT_PARAMS_CLASSIFICATION.get(model_name, {})
    else:
        if model_name not in REGRESSION_MODELS:
            raise ValueError(f"Unknown regression model: {model_name}")
        model_class = REGRESSION_MODELS[model_name]
        default_params = DEFAULT_PARAMS_REGRESSION.get(model_name, {})

    # 合并参数
    if params is None:
        params = default_params
    else:
        params = {**default_params, **params}

    # 创建模型
    model = model_class(**params)

    # 训练模型
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()

    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, n_jobs=-1)

    info = {
        'model_name': model_name,
        'task_type': task_type,
        'params': params,
        'training_time': training_time,
        'cv_scores': cv_scores.tolist(),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_samples': len(X_train),
        'n_features': X_train.shape[1]
    }

    logger.info(f"Model trained: CV score = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, info


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    task_type: str = 'classification',
    search_method: str = 'grid',
    param_grid: Optional[Dict] = None,
    n_iter: int = 50,
    cv_folds: int = 5
) -> Tuple[Any, Dict]:
    """
    超参数调优

    Args:
        X_train: 训练特征
        y_train: 训练标签
        model_name: 模型名称
        task_type: 'classification' or 'regression'
        search_method: 'grid' or 'random'
        param_grid: 参数网格（None使用默认）
        n_iter: 随机搜索迭代次数
        cv_folds: 交叉验证折数

    Returns:
        best_model: 最佳模型
        info: 包含最佳参数、CV分数、搜索历史的字典
    """
    logger.info(f"Hyperparameter tuning for {model_name} (method={search_method})")

    # 获取模型类和参数网格
    if task_type == 'classification':
        if model_name not in CLASSIFICATION_MODELS:
            raise ValueError(f"Unknown classification model: {model_name}")
        model_class = CLASSIFICATION_MODELS[model_name]
        default_param_grid = PARAM_GRIDS_CLASSIFICATION.get(model_name, {})
    else:
        if model_name not in REGRESSION_MODELS:
            raise ValueError(f"Unknown regression model: {model_name}")
        model_class = REGRESSION_MODELS[model_name]
        default_param_grid = PARAM_GRIDS_REGRESSION.get(model_name, {})

    # 使用默认参数网格或自定义参数网格
    if param_grid is None:
        param_grid = default_param_grid

    if not param_grid:
        logger.warning(f"No parameter grid defined for {model_name}, using default parameters")
        return train_single_model(X_train, y_train, model_name, task_type, cv_folds=cv_folds)

    # 创建基础模型
    base_model = model_class()

    # 执行搜索
    start_time = datetime.now()
    if search_method == 'grid':
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            n_jobs=-1,
            verbose=0
        )
    elif search_method == 'random':
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown search method: {search_method}")

    search.fit(X_train, y_train)
    tuning_time = (datetime.now() - start_time).total_seconds()

    best_model = search.best_estimator_

    info = {
        'model_name': model_name,
        'task_type': task_type,
        'search_method': search_method,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'tuning_time': tuning_time,
        'n_samples': len(X_train),
        'n_features': X_train.shape[1],
        'cv_results': {
            'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': search.cv_results_['std_test_score'].tolist(),
            'params': search.cv_results_['params']
        }
    }

    logger.info(f"Hyperparameter tuning completed: best score = {search.best_score_:.4f}")
    logger.info(f"Best parameters: {search.best_params_}")

    return best_model, info


def train_multiple_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = 'classification',
    models_to_try: Optional[List[str]] = None,
    cv_folds: int = 5
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    训练多个模型并比较性能

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        task_type: 'classification' or 'regression'
        models_to_try: 要尝试的模型列表（None表示全部）
        cv_folds: 交叉验证折数

    Returns:
        models_dict: {model_name: trained_model}字典
        comparison_df: 模型对比结果DataFrame
    """
    logger.info(f"Training multiple models (task_type={task_type})")

    # 确定要训练的模型
    if models_to_try is None:
        if task_type == 'classification':
            models_to_try = list(CLASSIFICATION_MODELS.keys())
        else:
            models_to_try = list(REGRESSION_MODELS.keys())

    models_dict = {}
    cv_scores_dict = {}  # 保存每个模型的交叉验证分数
    results = []

    for model_name in models_to_try:
        try:
            logger.info(f"Training {model_name}...")

            # 训练模型
            model, train_info = train_single_model(
                X_train, y_train, model_name, task_type, cv_folds=cv_folds
            )

            # 在测试集上评估
            y_pred = model.predict(X_test)

            if task_type == 'classification':
                from sklearn.metrics import accuracy_score, f1_score
                test_score = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                result = {
                    'model': model_name,
                    'cv_mean': train_info['cv_mean'],
                    'cv_std': train_info['cv_std'],
                    'test_accuracy': test_score,
                    'test_f1': f1,
                    'training_time': train_info['training_time']
                }
            else:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                test_r2 = r2_score(y_test, y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_mae = mean_absolute_error(y_test, y_pred)

                result = {
                    'model': model_name,
                    'cv_mean': train_info['cv_mean'],
                    'cv_std': train_info['cv_std'],
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'training_time': train_info['training_time']
                }

            results.append(result)
            models_dict[model_name] = model
            cv_scores_dict[model_name] = np.array(train_info['cv_scores'])  # 保存交叉验证分数

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {str(e)}")
            continue

    # 创建对比DataFrame
    comparison_df = pd.DataFrame(results)

    # 排序
    if task_type == 'classification':
        comparison_df = comparison_df.sort_values('test_accuracy', ascending=False)
    else:
        comparison_df = comparison_df.sort_values('test_r2', ascending=False)

    logger.info(f"Trained {len(models_dict)} models successfully")

    return models_dict, comparison_df, cv_scores_dict


# ============================================================================
# MODEL EVALUATION FUNCTIONS
# ============================================================================

def evaluate_classification(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    分类模型评估

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        class_names: 类别名称列表

    Returns:
        metrics: 包含accuracy, precision, recall, f1, roc_auc, confusion_matrix的字典
    """
    logger.info("Evaluating classification model")

    y_pred = model.predict(X_test)

    # 基础指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ROC AUC（如果模型支持概率预测）
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            n_classes = y_proba.shape[1]
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = None
    except Exception as e:
        logger.warning(f"Could not compute ROC AUC: {str(e)}")
        roc_auc = None

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 分类报告
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'n_samples': len(y_test),
        'n_classes': len(np.unique(y_test))
    }

    logger.info(f"Classification metrics: accuracy={accuracy:.4f}, f1={f1:.4f}")

    return metrics


def evaluate_regression(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    回归模型评估

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签

    Returns:
        metrics: 包含MSE, RMSE, MAE, R2, adjusted_R2的字典
    """
    logger.info("Evaluating regression model")

    y_pred = model.predict(X_test)

    # 计算指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 计算调整R²
    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # MAPE（平均绝对百分比误差）
    try:
        mape = mean_absolute_percentage_error(y_test, y_pred)
    except:
        # 如果y_test中有0值，MAPE会失败
        mape = None

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'adjusted_r2': adjusted_r2,
        'mape': mape,
        'n_samples': len(y_test)
    }

    logger.info(f"Regression metrics: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    return metrics


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> plt.Figure:
    """
    绘制特征重要性图

    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        top_n: 显示前N个特征

    Returns:
        fig: matplotlib Figure对象
    """
    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        logger.warning("Model does not have feature_importances_ or coef_ attribute")
        return None

    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    colors = sns.color_palette("viridis", len(importance_df))
    ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    绘制混淆矩阵热图

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    Returns:
        fig: matplotlib Figure对象
    """
    cm = confusion_matrix(y_true, y_pred)

    # 如果没有提供类别名称，自动从数据中提取
    if class_names is None:
        class_names = sorted(set(y_true) | set(y_pred))
        class_names = [str(c) for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    return fig


def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> plt.Figure:
    """
    绘制多个模型的ROC曲线对比

    Args:
        models_dict: {model_name: model_object}
        X_test: 测试特征
        y_test: 测试标签

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    colors = sns.color_palette("husl", len(models_dict))

    for (model_name, model), color in zip(models_dict.items(), colors):
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)

                # 二分类
                if y_proba.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=color, lw=2,
                           label=f'{model_name} (AUC = {roc_auc:.3f})')
        except Exception as e:
            logger.warning(f"Could not plot ROC curve for {model_name}: {str(e)}")
            continue

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> plt.Figure:
    """
    绘制预测值vs实际值散点图（回归）

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.5, s=50, edgecolors='k', linewidths=0.5)

    # 理想线（y=x）
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    # 计算R²
    r2 = r2_score(y_true, y_pred)

    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'Prediction vs Actual (R² = {r2:.4f})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> plt.Figure:
    """
    绘制残差图（回归）

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        fig: matplotlib Figure对象
    """
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # 残差散点图
    ax1.scatter(y_pred, residuals, alpha=0.5, s=50, edgecolors='k', linewidths=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 残差直方图
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig


def plot_learning_curves(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5
) -> plt.Figure:
    """
    绘制学习曲线

    显示训练集和验证集的性能随样本数量变化

    Args:
        model: 训练好的模型
        X: 特征数组
        y: 标签数组
        cv_folds: 交叉验证折数

    Returns:
        fig: matplotlib Figure对象
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv_folds, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='blue')

    ax.plot(train_sizes, val_mean, 'o-', color='green', label='Validation Score', linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='green')

    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_cv_scores(
    cv_scores: np.ndarray,
    metric_name: str = 'Score'
) -> plt.Figure:
    """
    绘制交叉验证分数分布

    Args:
        cv_scores: 交叉验证分数数组
        metric_name: 指标名称

    Returns:
        fig: matplotlib Figure对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # 条形图
    ax1.bar(range(1, len(cv_scores) + 1), cv_scores, color='steelblue', edgecolor='black')
    ax1.axhline(y=cv_scores.mean(), color='r', linestyle='--', lw=2, label=f'Mean = {cv_scores.mean():.4f}')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel(metric_name, fontsize=12)
    ax1.set_title('Cross-Validation Scores by Fold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 箱线图
    ax2.boxplot([cv_scores], labels=['CV Scores'], widths=0.5)
    ax2.set_ylabel(metric_name, fontsize=12)
    ax2.set_title(f'CV Score Distribution\nMean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig


# ============================================================================
# MODEL PERSISTENCE FUNCTIONS
# ============================================================================

def save_model(
    model: Any,
    filepath: str,
    metadata: Optional[Dict] = None,
    feature_names: Optional[List[str]] = None,
    scaler: Optional[Any] = None,
    encoder: Optional[Any] = None,
    task_type: Optional[str] = None
) -> bool:
    """
    保存模型及相关信息

    Args:
        model: 训练好的模型
        filepath: 保存路径（.pkl或.joblib）
        metadata: 元数据（性能指标、训练时间等）
        feature_names: 特征名称列表
        scaler: 特征缩放器
        encoder: 编码器
        task_type: 'classification' or 'regression'

    Returns:
        success: 是否成功保存
    """
    try:
        logger.info(f"Saving model to {filepath}")

        # 创建模型包
        model_package = {
            'model': model,
            'task_type': task_type,
            'feature_names': feature_names,
            'scaler': scaler,
            'encoder': encoder,
            'metadata': metadata or {},
            'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # 保存
        joblib.dump(model_package, filepath, compress=3)

        logger.info(f"Model saved successfully to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        return False


def load_model(filepath: str) -> Tuple[Any, Dict]:
    """
    加载模型及元数据

    Args:
        filepath: 模型文件路径

    Returns:
        model: 加载的模型
        model_package: 完整的模型包字典
    """
    try:
        logger.info(f"Loading model from {filepath}")

        model_package = joblib.load(filepath)

        model = model_package.get('model')
        if model is None:
            raise ValueError("Model not found in package")

        logger.info(f"Model loaded successfully from {filepath}")

        return model, model_package

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def get_model_info(filepath: str) -> Dict:
    """
    获取模型信息（不加载模型本身）

    Args:
        filepath: 模型文件路径

    Returns:
        info: 模型信息字典
    """
    try:
        model_package = joblib.load(filepath)

        info = {
            'task_type': model_package.get('task_type'),
            'feature_names': model_package.get('feature_names'),
            'n_features': len(model_package.get('feature_names', [])),
            'has_scaler': model_package.get('scaler') is not None,
            'has_encoder': model_package.get('encoder') is not None,
            'save_date': model_package.get('save_date'),
            'metadata': model_package.get('metadata', {})
        }

        return info

    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise


# ============================================================================
# HIGH-LEVEL API FUNCTIONS
# ============================================================================

def train_supervised_model(
    data_df: pd.DataFrame,
    target_column: Optional[str] = None,
    task_type: str = 'auto',
    test_size: float = 0.2,
    model_name: str = 'random_forest',
    feature_selection: Optional[str] = None,
    feature_scaling: str = 'standard',
    hyperparameter_tuning: bool = False,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Any, Dict]:
    """
    训练监督学习模型（高级API）

    Args:
        data_df: 包含特征和目标的DataFrame（最后一列为目标）
        target_column: 目标列名（None表示最后一列）
        task_type: 'classification', 'regression', 'auto'（自动检测）
        test_size: 测试集比例
        model_name: 模型名称
        feature_selection: 特征选择方法 (None, 'correlation', 'mutual_info', 'rfe', 'tree_based')
        feature_scaling: 特征缩放方法 ('standard', 'minmax', 'robust', None)
        hyperparameter_tuning: 是否进行超参数调优
        cv_folds: 交叉验证折数
        random_state: 随机种子

    Returns:
        model: 训练好的模型
        results: 包含评估指标、可视化、元数据的完整结果字典
    """
    logger.info("Starting supervised learning pipeline")

    # 确定目标列
    if target_column is None:
        target_column = data_df.columns[-1]
        logger.info(f"Using last column as target: {target_column}")

    # 分离特征和目标
    X = data_df.drop(columns=[target_column])
    y = data_df[target_column]

    # 自动检测任务类型
    if task_type == 'auto':
        n_unique = y.nunique()
        if n_unique <= 20:
            task_type = 'classification'
            logger.info(f"Auto-detected task type: classification ({n_unique} classes)")
        else:
            task_type = 'regression'
            logger.info(f"Auto-detected task type: regression")

    # 特征选择
    if feature_selection is not None:
        logger.info(f"Applying feature selection: {feature_selection}")
        if feature_selection == 'correlation':
            X, fs_info = select_features_correlation(X, y, threshold=0.1)
        elif feature_selection == 'mutual_info':
            X, fs_info = select_features_mutual_info(X, y, task_type=task_type)
        elif feature_selection == 'rfe':
            X, fs_info = select_features_rfe(X, y, task_type=task_type)
        elif feature_selection == 'tree_based':
            X, fs_info = select_features_tree_based(X, y, task_type=task_type)
        else:
            fs_info = None
    else:
        fs_info = None

    feature_names = X.columns.tolist()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # 特征缩放
    scaler = None
    if feature_scaling is not None:
        X_train_scaled, scaler = scale_features(X_train, method=feature_scaling, fit=True)
        X_test_scaled, _ = scale_features(X_test, method=feature_scaling, scaler=scaler, fit=False)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    # 训练模型
    if hyperparameter_tuning:
        logger.info("Training with hyperparameter tuning")
        model, train_info = hyperparameter_tuning(
            X_train_scaled, y_train, model_name, task_type, cv_folds=cv_folds
        )
    else:
        logger.info("Training with default parameters")
        model, train_info = train_single_model(
            X_train_scaled, y_train, model_name, task_type, cv_folds=cv_folds
        )

    # 评估模型
    if task_type == 'classification':
        metrics = evaluate_classification(model, X_test_scaled, y_test)
    else:
        metrics = evaluate_regression(model, X_test_scaled, y_test)

    # 预测
    y_pred = model.predict(X_test_scaled)

    # 计算特征统计信息（用于虚拟样本生成）
    feature_stats = {}
    for col in feature_names:
        col_data = X[col]
        n_unique = col_data.nunique()
        # 判断是连续值还是离散值（唯一值少于10个或少于样本数的5%视为离散值）
        is_categorical = n_unique <= 10 or n_unique < len(col_data) * 0.05

        if is_categorical:
            feature_stats[col] = {
                'type': 'categorical',
                'unique_values': sorted(col_data.unique().tolist()),
                'n_unique': n_unique
            }
        else:
            feature_stats[col] = {
                'type': 'continuous',
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'n_unique': n_unique
            }

    # 组装结果
    results = {
        'model': model,
        'task_type': task_type,
        'model_name': model_name,
        'feature_names': feature_names,
        'feature_stats': feature_stats,
        'scaler': scaler,
        'metrics': metrics,
        'predictions': y_pred,
        'y_test': y_test,
        'train_info': train_info,
        'feature_selection_info': fs_info,
        'n_features': len(feature_names),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test)
    }

    logger.info("Supervised learning pipeline completed successfully")

    return model, results


def compare_models_automl(
    data_df: pd.DataFrame,
    target_column: Optional[str] = None,
    task_type: str = 'auto',
    models_to_compare: Optional[List[str]] = None,
    test_size: float = 0.2,
    feature_selection: Optional[str] = None,
    feature_scaling: str = 'standard',
    cv_folds: int = 5,
    tune_best: bool = True,
    random_state: int = 42
) -> Tuple[Any, pd.DataFrame, Dict]:
    """
    AutoML流程 - 自动比较多个模型并选择最佳

    Args:
        data_df: 包含特征和目标的DataFrame
        target_column: 目标列名（None表示最后一列）
        task_type: 'classification', 'regression', 'auto'
        models_to_compare: 要对比的模型列表（None表示全部）
        test_size: 测试集比例
        feature_selection: 特征选择方法
        feature_scaling: 特征缩放方法
        cv_folds: 交叉验证折数
        tune_best: 是否对最佳模型进行超参数调优
        random_state: 随机种子

    Returns:
        best_model: 最佳模型
        comparison_df: 模型对比结果DataFrame
        all_results: 包含所有模型详细结果的字典
    """
    logger.info("Starting AutoML pipeline")

    # 确定目标列
    if target_column is None:
        target_column = data_df.columns[-1]
        logger.info(f"Using last column as target: {target_column}")

    # 分离特征和目标
    X = data_df.drop(columns=[target_column])
    y = data_df[target_column]

    # 自动检测任务类型
    if task_type == 'auto':
        n_unique = y.nunique()
        if n_unique <= 20:
            task_type = 'classification'
            logger.info(f"Auto-detected task type: classification ({n_unique} classes)")
        else:
            task_type = 'regression'
            logger.info(f"Auto-detected task type: regression")

    # 特征选择
    if feature_selection is not None:
        logger.info(f"Applying feature selection: {feature_selection}")
        if feature_selection == 'correlation':
            X, fs_info = select_features_correlation(X, y, threshold=0.1)
        elif feature_selection == 'mutual_info':
            X, fs_info = select_features_mutual_info(X, y, task_type=task_type)
        elif feature_selection == 'rfe':
            X, fs_info = select_features_rfe(X, y, task_type=task_type)
        elif feature_selection == 'tree_based':
            X, fs_info = select_features_tree_based(X, y, task_type=task_type)
    else:
        fs_info = None

    feature_names = X.columns.tolist()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # 特征缩放
    scaler = None
    if feature_scaling is not None:
        X_train_scaled, scaler = scale_features(X_train, method=feature_scaling, fit=True)
        X_test_scaled, _ = scale_features(X_test, method=feature_scaling, scaler=scaler, fit=False)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    # 训练多个模型
    models_dict, comparison_df, cv_scores_dict = train_multiple_models(
        X_train_scaled, y_train, X_test_scaled, y_test,
        task_type=task_type, models_to_try=models_to_compare, cv_folds=cv_folds
    )

    # 选择最佳模型
    if task_type == 'classification':
        best_model_name = comparison_df.iloc[0]['model']
    else:
        best_model_name = comparison_df.iloc[0]['model']

    best_model = models_dict[best_model_name]

    logger.info(f"Best model: {best_model_name}")

    # 对最佳模型进行超参数调优
    if tune_best:
        logger.info(f"Tuning best model: {best_model_name}")
        best_model, tune_info = hyperparameter_tuning(
            X_train_scaled, y_train, best_model_name, task_type, cv_folds=cv_folds
        )
    else:
        tune_info = None

    # 评估最佳模型
    if task_type == 'classification':
        metrics = evaluate_classification(best_model, X_test_scaled, y_test)
    else:
        metrics = evaluate_regression(best_model, X_test_scaled, y_test)

    # 生成预测结果
    predictions = best_model.predict(X_test_scaled)

    # 计算特征统计信息（用于虚拟样本生成）
    feature_stats = {}
    for col in feature_names:
        col_data = X[col]
        n_unique = col_data.nunique()
        is_categorical = n_unique <= 10 or n_unique < len(col_data) * 0.05

        if is_categorical:
            feature_stats[col] = {
                'type': 'categorical',
                'unique_values': sorted(col_data.unique().tolist()),
                'n_unique': n_unique
            }
        else:
            feature_stats[col] = {
                'type': 'continuous',
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'n_unique': n_unique
            }

    # 组装结果
    all_results = {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'task_type': task_type,
        'feature_names': feature_names,
        'feature_stats': feature_stats,
        'scaler': scaler,
        'metrics': metrics,
        'comparison_df': comparison_df,
        'all_models': models_dict,
        'tune_info': tune_info,
        'feature_selection_info': fs_info,
        'n_features': len(feature_names),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'y_test': y_test,
        'predictions': predictions,
        'cv_scores_dict': cv_scores_dict,
        'X_train_scaled': X_train_scaled
    }

    logger.info("AutoML pipeline completed successfully")

    return best_model, comparison_df, all_results


def plot_shap_analysis(
    model: Any,
    X_data: np.ndarray,
    feature_names: List[str],
    max_display: int = 20
) -> plt.Figure:
    """
    生成SHAP分析可视化

    Args:
        model: 训练好的模型
        X_data: 特征数据（用于计算SHAP值）
        feature_names: 特征名称列表
        max_display: 最多显示的特征数量

    Returns:
        fig: matplotlib Figure对象
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP library not installed. Please install with: pip install shap")
        # 返回一个提示图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'SHAP library not installed\nPlease install with: pip install shap',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    try:
        # 创建SHAP explainer
        explainer = shap.Explainer(model, X_data)
        shap_values = explainer(X_data)

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        # 生成summary plot
        shap.summary_plot(shap_values, X_data, feature_names=feature_names,
                         max_display=max_display, show=False)

        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Failed to generate SHAP analysis: {str(e)}")
        # 返回错误提示图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Failed to generate SHAP analysis\nError: {str(e)}',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig


def plot_cv_scores(
    cv_scores_dict: Dict[str, np.ndarray],
    metric_name: str = 'Score'
) -> plt.Figure:
    """
    可视化交叉验证每一折的成绩

    Args:
        cv_scores_dict: 字典，键为模型名称，值为交叉验证分数数组
        metric_name: 指标名称

    Returns:
        fig: matplotlib Figure对象
    """
    if not cv_scores_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No cross-validation scores available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    # 准备数据
    models = list(cv_scores_dict.keys())
    n_models = len(models)
    n_folds = len(cv_scores_dict[models[0]])

    # 设置x轴位置
    x = np.arange(n_folds)
    width = 0.8 / n_models

    # 为每个模型绘制柱状图
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))

    for i, (model_name, scores) in enumerate(cv_scores_dict.items()):
        offset = (i - n_models/2) * width + width/2
        bars = ax.bar(x + offset, scores, width, label=model_name,
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

        # 在柱子上方显示数值
        for j, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontsize=8, rotation=0)

    # 设置图表属性
    ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'Cross-Validation {metric_name} by Fold', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    return fig

