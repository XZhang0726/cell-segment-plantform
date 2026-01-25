"""
统一的细胞分割接口

整合传统图像处理方法和深度学习方法
"""
import numpy as np
from pathlib import Path
from typing import Optional, Union, Literal
from enum import Enum

from ..core.segmentation.threshold import ThresholdSegmentation
from ..core.segmentation.watershed import WatershedSegmentation
from ..core.segmentation.edge import EdgeDetection
from ..core.segmentation.cellpose_seg import cellpose_segment
from ..inference.predictor import Predictor
from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class SegmentationMethod(Enum):
    """分割方法枚举"""
    OTSU = "otsu"
    ADAPTIVE = "adaptive"
    WATERSHED = "watershed"
    EDGE_CANNY = "edge_canny"
    DEEP_LEARNING = "deep_learning"
    CELLPOSE = "cellpose"


class CellSegmenter:
    """
    统一的细胞分割器

    提供传统方法和深度学习方法的统一接口
    """

    def __init__(
        self,
        method: Union[str, SegmentationMethod] = SegmentationMethod.OTSU,
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        初始化分割器

        Args:
            method: 分割方法 ("otsu", "adaptive", "watershed", "edge_canny", "deep_learning")
            model_path: 预训练模型路径（仅用于deep_learning方法）
            device: 运行设备（仅用于deep_learning方法）
        """
        if isinstance(method, str):
            method = SegmentationMethod(method)

        self.method = method
        self.model_path = model_path
        self.device = device

        # 初始化传统方法处理器
        self.threshold_seg = ThresholdSegmentation()
        self.watershed_seg = WatershedSegmentation()
        self.edge_detector = EdgeDetection()

        # 初始化深度学习预测器（如果需要）
        self.predictor = None
        if self.method == SegmentationMethod.DEEP_LEARNING:
            if model_path is None:
                logger.warning("Deep learning method selected but no model_path provided. "
                             "Using randomly initialized weights.")
            self.predictor = Predictor(model_path=model_path, device=device)
            logger.info("Initialized deep learning predictor")

        logger.info(f"CellSegmenter initialized with method: {self.method.value}")

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """将图像转换为灰度图"""
        import cv2
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def segment(
        self,
        image: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        对图像进行分割

        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            **kwargs: 方法特定的参数

        Returns:
            分割掩码 (H, W)
        """
        if self.method == SegmentationMethod.OTSU:
            return self._segment_otsu(image, **kwargs)
        elif self.method == SegmentationMethod.ADAPTIVE:
            return self._segment_adaptive(image, **kwargs)
        elif self.method == SegmentationMethod.WATERSHED:
            return self._segment_watershed(image, **kwargs)
        elif self.method == SegmentationMethod.EDGE_CANNY:
            return self._segment_edge(image, **kwargs)
        elif self.method == SegmentationMethod.CELLPOSE:
            return self._segment_cellpose(image, **kwargs)
        elif self.method == SegmentationMethod.DEEP_LEARNING:
            return self._segment_deep_learning(image, **kwargs)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")

    def _segment_otsu(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """使用Otsu阈值分割"""
        gray = self._to_grayscale(image)
        return self.threshold_seg.otsu_threshold(gray)

    def _segment_adaptive(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """使用自适应阈值分割"""
        gray = self._to_grayscale(image)
        block_size = kwargs.get('block_size', 11)
        C = kwargs.get('C', 2)
        return self.threshold_seg.adaptive_threshold(gray, block_size=block_size, C=C)

    def _segment_watershed(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """使用分水岭算法分割"""
        # 转换为灰度图
        gray = self._to_grayscale(image)
        # 二值化
        binary = self.threshold_seg.otsu_threshold(gray)
        # 应用分水岭算法
        return self.watershed_seg.watershed_distance_transform(binary)

    def _segment_edge(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """使用边缘检测分割"""
        gray = self._to_grayscale(image)
        low_threshold = kwargs.get('low_threshold', 50)
        high_threshold = kwargs.get('high_threshold', 150)
        return self.edge_detector.canny(gray, threshold1=low_threshold, threshold2=high_threshold)

    def _segment_deep_learning(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """使用深度学习模型分割"""
        if self.predictor is None:
            raise RuntimeError("Deep learning predictor not initialized. "
                             "Please provide model_path when creating CellSegmenter.")

        target_size = kwargs.get('target_size', (256, 256))
        threshold = kwargs.get('threshold', 0.5)
        return_original_size = kwargs.get('return_original_size', True)

        return self.predictor.predict(
            image,
            target_size=target_size,
            threshold=threshold,
            return_original_size=return_original_size
        )

    def _segment_cellpose(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """使用Cellpose深度学习模型分割"""
        model_type = kwargs.get('model_type', 'cyto2')
        diameter = kwargs.get('diameter', None)
        channels = kwargs.get('channels', None)
        progress_bar = kwargs.get('progress_bar', None)
        use_gpu = kwargs.get('use_gpu', False)

        return cellpose_segment(
            image,
            model_type=model_type,
            diameter=diameter,
            channels=channels,
            progress_bar=progress_bar,
            use_gpu=use_gpu
        )
