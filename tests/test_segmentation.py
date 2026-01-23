"""
传统分割算法的单元测试
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.core.segmentation.threshold import ThresholdSegmentation
from src.core.segmentation.edge import EdgeDetection
from src.core.segmentation.morphology import MorphologicalOps
from src.core.segmentation.watershed import WatershedSegmentation
from src.core.segmentation.contour import ContourAnalysis


class TestThresholdSegmentation:
    """测试阈值分割算法"""

    @pytest.fixture
    def sample_gray_image(self):
        """创建测试用的灰度图像"""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        return image

    def test_otsu_threshold(self, sample_gray_image):
        """测试Otsu阈值分割"""
        binary = ThresholdSegmentation.otsu_threshold(sample_gray_image)

        # 验证输出是二值图像
        assert binary.shape == sample_gray_image.shape
        assert set(np.unique(binary)).issubset({0, 255})

    def test_otsu_threshold_return_value(self, sample_gray_image):
        """测试Otsu阈值分割返回阈值"""
        binary, threshold = ThresholdSegmentation.otsu_threshold(
            sample_gray_image, return_threshold=True
        )

        # 验证返回了阈值
        assert isinstance(threshold, (int, float))
        assert 0 <= threshold <= 255

    def test_fixed_threshold(self, sample_gray_image):
        """测试固定阈值分割"""
        binary = ThresholdSegmentation.fixed_threshold(
            sample_gray_image, threshold=127
        )

        # 验证输出是二值图像
        assert binary.shape == sample_gray_image.shape
        assert set(np.unique(binary)).issubset({0, 255})

    def test_adaptive_threshold(self, sample_gray_image):
        """测试自适应阈值分割"""
        binary = ThresholdSegmentation.adaptive_threshold(
            sample_gray_image, block_size=11
        )

        # 验证输出是二值图像
        assert binary.shape == sample_gray_image.shape
        assert set(np.unique(binary)).issubset({0, 255})

    def test_threshold_invalid_input(self):
        """测试无效输入"""
        # 彩色图像应该抛出错误
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            ThresholdSegmentation.otsu_threshold(color_image)


class TestEdgeDetection:
    """测试边缘检测算法"""

    @pytest.fixture
    def sample_gray_image(self):
        """创建测试用的灰度图像"""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        return image

    def test_canny_edge(self, sample_gray_image):
        """测试Canny边缘检测"""
        edges = EdgeDetection.canny(sample_gray_image, threshold1=50, threshold2=150)

        # 验证输出形状
        assert edges.shape == sample_gray_image.shape
        # 验证是二值图像
        assert set(np.unique(edges)).issubset({0, 255})

    def test_sobel_edge(self, sample_gray_image):
        """测试Sobel边缘检测"""
        edges = EdgeDetection.sobel(sample_gray_image, dx=1, dy=1, ksize=3)

        # 验证输出形状
        assert edges.shape == sample_gray_image.shape
        assert edges.dtype == np.uint8

    def test_laplacian_edge(self, sample_gray_image):
        """测试Laplacian边缘检测"""
        edges = EdgeDetection.laplacian(sample_gray_image, ksize=3)

        # 验证输出形状
        assert edges.shape == sample_gray_image.shape
        assert edges.dtype == np.uint8

    def test_scharr_edge(self, sample_gray_image):
        """测试Scharr边缘检测"""
        edges = EdgeDetection.scharr(sample_gray_image, dx=1, dy=0)

        # 验证输出形状
        assert edges.shape == sample_gray_image.shape
        assert edges.dtype == np.uint8

    def test_edge_invalid_input(self):
        """测试无效输入"""
        # 彩色图像应该抛出错误
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            EdgeDetection.canny(color_image)


class TestMorphologicalOps:
    """测试形态学操作"""

    @pytest.fixture
    def sample_binary_image(self):
        """创建测试用的二值图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[30:70, 30:70] = 255
        return image

    def test_erode(self, sample_binary_image):
        """测试腐蚀操作"""
        eroded = MorphologicalOps.erode(sample_binary_image, kernel_size=(5, 5))

        # 验证输出形状
        assert eroded.shape == sample_binary_image.shape
        # 腐蚀后白色区域应该减少
        assert np.sum(eroded) < np.sum(sample_binary_image)

    def test_dilate(self, sample_binary_image):
        """测试膨胀操作"""
        dilated = MorphologicalOps.dilate(sample_binary_image, kernel_size=(5, 5))

        # 验证输出形状
        assert dilated.shape == sample_binary_image.shape
        # 膨胀后白色区域应该增加
        assert np.sum(dilated) > np.sum(sample_binary_image)

    def test_opening(self, sample_binary_image):
        """测试开运算"""
        opened = MorphologicalOps.opening(sample_binary_image, kernel_size=(5, 5))

        # 验证输出形状
        assert opened.shape == sample_binary_image.shape

    def test_closing(self, sample_binary_image):
        """测试闭运算"""
        closed = MorphologicalOps.closing(sample_binary_image, kernel_size=(5, 5))

        # 验证输出形状
        assert closed.shape == sample_binary_image.shape

    def test_gradient(self, sample_binary_image):
        """测试形态学梯度"""
        gradient = MorphologicalOps.gradient(sample_binary_image, kernel_size=(5, 5))

        # 验证输出形状
        assert gradient.shape == sample_binary_image.shape


class TestWatershedSegmentation:
    """测试分水岭分割算法"""

    @pytest.fixture
    def sample_binary_image(self):
        """创建测试用的二值图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        # 创建两个分离的圆形
        import cv2
        cv2.circle(image, (30, 30), 15, 255, -1)
        cv2.circle(image, (70, 70), 15, 255, -1)
        return image

    @pytest.fixture
    def sample_color_image(self):
        """创建测试用的彩色图像"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return image

    def test_watershed_distance_transform(self, sample_binary_image):
        """测试基于距离变换的分水岭分割"""
        labels = WatershedSegmentation.watershed_distance_transform(
            sample_binary_image, min_distance=10
        )

        # 验证输出形状
        assert labels.shape == sample_binary_image.shape
        # 应该有多个区域
        assert len(np.unique(labels)) > 1

    def test_watershed_marker_controlled(self, sample_color_image, sample_binary_image):
        """测试标记控制的分水岭分割"""
        markers = WatershedSegmentation.watershed_marker_controlled(
            sample_color_image, sample_binary_image
        )

        # 验证输出形状
        assert markers.shape == sample_color_image.shape[:2]


class TestContourAnalysis:
    """测试轮廓检测和分析"""

    @pytest.fixture
    def sample_binary_image(self):
        """创建测试用的二值图像"""
        image = np.zeros((100, 100), dtype=np.uint8)
        # 创建几个矩形
        import cv2
        cv2.rectangle(image, (10, 10), (30, 30), 255, -1)
        cv2.rectangle(image, (50, 50), (80, 80), 255, -1)
        return image

    def test_find_contours(self, sample_binary_image):
        """测试查找轮廓"""
        contours = ContourAnalysis.find_contours(sample_binary_image)

        # 应该找到至少一个轮廓
        assert len(contours) > 0
        # 每个轮廓应该是numpy数组
        assert all(isinstance(c, np.ndarray) for c in contours)

    def test_filter_contours(self, sample_binary_image):
        """测试过滤轮廓"""
        contours = ContourAnalysis.find_contours(sample_binary_image)
        filtered = ContourAnalysis.filter_contours(contours, min_area=100)

        # 过滤后的轮廓数量应该小于或等于原始数量
        assert len(filtered) <= len(contours)

    def test_get_contour_properties(self, sample_binary_image):
        """测试获取轮廓属性"""
        contours = ContourAnalysis.find_contours(sample_binary_image)
        if len(contours) > 0:
            properties = ContourAnalysis.get_contour_properties(contours[0])

            # 验证返回的属性
            assert 'area' in properties
            assert 'perimeter' in properties
            assert 'circularity' in properties
            assert 'aspect_ratio' in properties
            assert properties['area'] >= 0

    def test_draw_contours(self, sample_binary_image):
        """测试绘制轮廓"""
        contours = ContourAnalysis.find_contours(sample_binary_image)
        result = ContourAnalysis.draw_contours(
            sample_binary_image, contours, color=(0, 255, 0)
        )

        # 验证输出是彩色图像
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_get_bounding_boxes(self, sample_binary_image):
        """测试获取边界框"""
        contours = ContourAnalysis.find_contours(sample_binary_image)
        boxes = ContourAnalysis.get_bounding_boxes(contours)

        # 边界框数量应该等于轮廓数量
        assert len(boxes) == len(contours)

