"""
配置管理模块

提供配置文件的加载和管理功能
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from .paths import get_configs_dir


class Config:
    """配置管理类"""

    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        初始化配置

        Args:
            config_dict: 配置字典
        """
        self._config = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，支持点号分隔的嵌套键，如'model.name'
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        设置配置值

        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()

    @classmethod
    def from_yaml(cls, yaml_file: Path) -> 'Config':
        """
        从YAML文件加载配置

        Args:
            yaml_file: YAML文件路径

        Returns:
            Config实例
        """
        yaml_file = Path(yaml_file)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_file}")

        with open(yaml_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    def save_yaml(self, yaml_file: Path):
        """
        保存配置到YAML文件

        Args:
            yaml_file: YAML文件路径
        """
        yaml_file = Path(yaml_file)
        yaml_file.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)


def load_config(config_name: str) -> Config:
    """
    加载配置文件

    Args:
        config_name: 配置文件名（不含扩展名）或完整路径

    Returns:
        Config实例
    """
    config_path = Path(config_name)

    # 如果是完整路径
    if config_path.exists():
        return Config.from_yaml(config_path)

    # 如果只是文件名，从configs目录加载
    config_path = get_configs_dir() / f"{config_name}.yaml"
    if config_path.exists():
        return Config.from_yaml(config_path)

    # 如果都找不到，返回空配置
    return Config()
