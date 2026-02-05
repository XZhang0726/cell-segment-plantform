"""
生成主动学习所需的CSV文件：
1. 初始训练数据（有标签）- 每类随机抽取100条，共400条
2. 未标注样本池（无标签）- 剩余约12,000条
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 设置随机种子，保证可复现
np.random.seed(42)

# 路径配置
BASE_DIR = Path(r"c:\Users\XB001\Desktop\cc_works\xibaofenge")
INPUT_FILE = BASE_DIR / "datasets" / "blood_cells_features" / "blood_cells_data.csv"
OUTPUT_DIR = BASE_DIR / "datasets" / "active_learning_data"

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 每类抽取的样本数
SAMPLES_PER_CLASS = 100

print("=" * 50)
print("主动学习数据生成脚本")
print("=" * 50)

# 读取原始数据
print(f"\n读取数据: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"总样本数: {len(df)}")
print(f"特征数: {len(df.columns) - 1}")  # 减去标签列
print(f"标签列: diagnosis")

# 查看类别分布
print("\n原始数据类别分布:")
print(df['diagnosis'].value_counts())

# 分层抽样：每类抽取相同数量的样本
train_indices = []
for label in df['diagnosis'].unique():
    label_indices = df[df['diagnosis'] == label].index.tolist()
    sampled = np.random.choice(label_indices, size=SAMPLES_PER_CLASS, replace=False)
    train_indices.extend(sampled)

# 创建训练集和未标注池
train_df = df.loc[train_indices].copy()
pool_indices = df.index.difference(train_indices)
pool_df = df.loc[pool_indices].copy()

# 未标注池移除标签列
pool_df_unlabeled = pool_df.drop(columns=['diagnosis'])

# 保存训练集（有标签）
train_output = OUTPUT_DIR / "initial_train_labeled.csv"
train_df.to_csv(train_output, index=False)
print(f"\n初始训练数据已保存: {train_output}")
print(f"  - 样本数: {len(train_df)}")
print(f"  - 类别分布:")
print(train_df['diagnosis'].value_counts().to_string())

# 保存未标注池（无标签）
pool_output = OUTPUT_DIR / "unlabeled_pool.csv"
pool_df_unlabeled.to_csv(pool_output, index=False)
print(f"\n未标注样本池已保存: {pool_output}")
print(f"  - 样本数: {len(pool_df_unlabeled)}")
print(f"  - 特征数: {len(pool_df_unlabeled.columns)}")

# 额外保存：未标注池的真实标签（用于模拟主动学习时的标注）
pool_labels_output = OUTPUT_DIR / "pool_true_labels.csv"
pool_df[['diagnosis']].to_csv(pool_labels_output, index=False)
print(f"\n未标注池真实标签已保存: {pool_labels_output}")
print(f"  (用于模拟主动学习时的标注查询)")

print("\n" + "=" * 50)
print("数据生成完成!")
print("=" * 50)
print(f"\n输出目录: {OUTPUT_DIR}")
print("\n生成的文件:")
print(f"  1. initial_train_labeled.csv  - 初始训练数据（有标签）")
print(f"  2. unlabeled_pool.csv         - 未标注样本池（无标签）")
print(f"  3. pool_true_labels.csv       - 未标注池真实标签（模拟用）")
