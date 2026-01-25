"""
训练配置管理模块

提供训练超参数的配置管理
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import yaml

from ..core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """训练配置类"""

    # 模型配置
    model_name: str = "unet"
    n_channels: int = 3
    n_classes: int = 1
    bilinear: bool = True

    # 训练配置
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # 数据配置
    image_size: tuple = (256, 256)
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # 损失函数配置
    loss_type: str = "bce_dice"  # "dice", "bce_dice", "focal"
    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # 优化器配置
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    momentum: float = 0.9

    # 学习率调度器配置
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # 其他配置
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    save_dir: str = "checkpoints"
    log_interval: int = 10

    def save(self, path: str):
        """保存配置到YAML文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

        logger.info(f"Saved config to {path}")

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """从YAML文件加载配置"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        logger.info(f"Loaded config from {path}")
        return cls(**config_dict)


def get_default_config() -> TrainingConfig:
    """获取默认配置"""
    return TrainingConfig()

