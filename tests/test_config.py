"""
训练配置单元测试
"""
import pytest
import yaml
import sys
from pathlib import Path
import tempfile

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.config import TrainingConfig, get_default_config


class TestTrainingConfig:
    """测试训练配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = TrainingConfig()

        # 检查默认值
        assert config.model_name == "unet"
        assert config.n_channels == 3
        assert config.n_classes == 1
        assert config.epochs == 100
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4

    def test_custom_config(self):
        """测试自定义配置"""
        config = TrainingConfig(
            model_name="custom_unet",
            epochs=50,
            batch_size=16,
            learning_rate=1e-3
        )

        assert config.model_name == "custom_unet"
        assert config.epochs == 50
        assert config.batch_size == 16
        assert config.learning_rate == 1e-3

    def test_save_config(self):
        """测试保存配置"""
        config = TrainingConfig(epochs=50, batch_size=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config.save(str(config_path))

            # 检查文件是否存在
            assert config_path.exists()

            # 检查文件内容
            with open(config_path, 'r', encoding='utf-8') as f:
                saved_data = yaml.safe_load(f)

            assert saved_data['epochs'] == 50
            assert saved_data['batch_size'] == 16

    def test_load_config(self):
        """测试加载配置"""
        config = TrainingConfig(epochs=50, batch_size=16, learning_rate=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config.save(str(config_path))

            # 加载配置
            loaded_config = TrainingConfig.load(str(config_path))

            # 检查加载的配置是否正确
            assert loaded_config.epochs == 50
            assert loaded_config.batch_size == 16
            assert loaded_config.learning_rate == 1e-3

    def test_load_nonexistent_config(self):
        """测试加载不存在的配置文件"""
        with pytest.raises(FileNotFoundError):
            TrainingConfig.load("nonexistent_config.yaml")


class TestGetDefaultConfig:
    """测试获取默认配置函数"""

    def test_get_default_config(self):
        """测试获取默认配置"""
        config = get_default_config()

        # 应该返回TrainingConfig实例
        assert isinstance(config, TrainingConfig)

        # 检查默认值
        assert config.model_name == "unet"
        assert config.epochs == 100
        assert config.batch_size == 8
