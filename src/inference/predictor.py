"""
模型推理模块

提供加载预训练模型并进行预测的功能
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import cv2

from ..core.models.unet import UNet
from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    预测器类

    用于加载预训练模型并对新图像进行分割预测
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        n_channels: int = 3,
        n_classes: int = 1,
        bilinear: bool = True
    ):
        """
        初始化预测器

        Args:
            model_path: 预训练模型路径(.pth文件)
            device: 运行设备 ("cuda" 或 "cpu")
            n_channels: 输入图像通道数
            n_classes: 输出类别数
            bilinear: 是否使用双线性插值上采样
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 创建模型
        self.model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            bilinear=bilinear
        )
        self.model.to(self.device)
        self.model.eval()

        # 加载预训练权重
        if model_path is not None:
            self.load_model(model_path)
            logger.info(f"Loaded pretrained model from {model_path}")
        else:
            logger.warning("No pretrained model loaded. Using randomly initialized weights.")

        logger.info(f"Predictor initialized on device: {self.device}")

    def load_model(self, model_path: str):
        """
        加载预训练模型权重

        Args:
            model_path: 模型权重文件路径
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)

        # 处理不同的保存格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 从训练检查点加载
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # 直接加载state_dict
            self.model.load_state_dict(checkpoint)
            logger.info("Loaded model state dict")

        self.model.eval()

    def preprocess(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        预处理图像

        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            target_size: 目标尺寸 (height, width)，如果为None则保持原尺寸

        Returns:
            预处理后的张量 (1, C, H, W)
        """
        # 调整大小
        if target_size is not None:
            image = cv2.resize(image, (target_size[1], target_size[0]))

        # 归一化到[0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # 转换维度顺序
        if image.ndim == 2:
            # 灰度图 (H, W) -> (1, H, W)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:
            # 彩色图 (H, W, C) -> (C, H, W)
            image = np.transpose(image, (2, 0, 1))

        # 添加batch维度并转换为tensor
        image = torch.from_numpy(image.copy()).float()
        image = image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        return image

    def postprocess(
        self,
        output: torch.Tensor,
        threshold: float = 0.5,
        original_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        后处理模型输出

        Args:
            output: 模型输出张量 (1, C, H, W)
            threshold: 二值化阈值
            original_size: 原始图像尺寸 (height, width)，如果提供则调整回原尺寸

        Returns:
            二值化掩码 (H, W)
        """
        # 应用sigmoid并转换为numpy
        output = torch.sigmoid(output)
        mask = output.squeeze().cpu().numpy()  # (H, W)

        # 二值化
        mask = (mask > threshold).astype(np.uint8) * 255

        # 调整回原尺寸
        if original_size is not None:
            mask = cv2.resize(mask, (original_size[1], original_size[0]))

        return mask

    def predict(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = (256, 256),
        threshold: float = 0.5,
        return_original_size: bool = True
    ) -> np.ndarray:
        """
        对单张图像进行分割预测

        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            target_size: 模型输入尺寸 (height, width)
            threshold: 二值化阈值
            return_original_size: 是否将结果调整回原始尺寸

        Returns:
            预测的二值化掩码 (H, W)
        """
        original_size = image.shape[:2] if return_original_size else None

        # 预处理
        input_tensor = self.preprocess(image, target_size)
        input_tensor = input_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)

        # 后处理
        mask = self.postprocess(output, threshold, original_size)

        return mask


