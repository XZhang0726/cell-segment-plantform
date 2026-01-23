"""
图像IO功能的单元测试
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.data.image_io import ImageIO, load_image, save_image


class TestImageIO:
    """测试ImageIO类"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_image(self):
        """创建测试用的样本图像"""
        # 创建一个简单的RGB图像
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return image

    @pytest.fixture
    def sample_gray_image(self):
        """创建测试用的灰度图像"""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        return image

    def test_save_and_load_image(self, temp_dir, sample_image):
        """测试保存和加载图像"""
        # 保存图像
        image_path = temp_dir / "test_image.png"
        save_image(sample_image, image_path)

        # 验证文件存在
        assert image_path.exists()

        # 加载图像
        loaded_image = load_image(image_path)

        # 验证形状一致
        assert loaded_image.shape == sample_image.shape

    def test_load_grayscale(self, temp_dir, sample_image):
        """测试加载灰度图像"""
        # 保存彩色图像
        image_path = temp_dir / "test_color.png"
        save_image(sample_image, image_path)

        # 以灰度模式加载
        gray_image = load_image(image_path, grayscale=True)

        # 验证是灰度图
        assert gray_image.ndim == 2

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_file.png")

    def test_load_unsupported_format(self, temp_dir):
        """测试加载不支持的格式"""
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.touch()

        with pytest.raises(ValueError):
            load_image(unsupported_file)

    def test_get_image_info(self, temp_dir, sample_image):
        """测试获取图像信息"""
        # 保存图像
        image_path = temp_dir / "test_info.png"
        save_image(sample_image, image_path)

        # 获取信息
        info = ImageIO.get_image_info(image_path)

        # 验证信息
        assert info['width'] == 100
        assert info['height'] == 100
        assert info['format'] == 'PNG'

    def test_load_images_batch(self, temp_dir, sample_image):
        """测试批量加载图像"""
        # 创建多个测试图像
        image_paths = []
        for i in range(3):
            path = temp_dir / f"test_{i}.png"
            save_image(sample_image, path)
            image_paths.append(path)

        # 批量加载
        images = ImageIO.load_images_batch(image_paths)

        # 验证加载数量
        assert len(images) == 3

        # 验证每个图像的形状
        for img in images:
            assert img.shape == sample_image.shape

    def test_different_backends(self, temp_dir, sample_image):
        """测试不同的后端"""
        backends = ['opencv', 'pillow', 'skimage']

        for backend in backends:
            # 保存图像
            image_path = temp_dir / f"test_{backend}.png"
            ImageIO.save_image(sample_image, image_path, backend=backend)

            # 加载图像
            loaded = ImageIO.load_image(image_path, backend=backend)

            # 验证形状
            assert loaded.shape == sample_image.shape
