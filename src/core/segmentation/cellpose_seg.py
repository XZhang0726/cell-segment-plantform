"""
Cellpose深度学习细胞分割模块

使用Cellpose预训练模型进行细胞分割
"""
import os
# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from loguru import logger
import multiprocessing

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except Exception as e:
    CELLPOSE_AVAILABLE = False
    logger.warning(f"Cellpose not available: {type(e).__name__}: {str(e)}")


class StreamlitProgressBar:
    """模拟QProgressBar接口，用于Cellpose进度回调"""
    def __init__(self, streamlit_progress_bar=None):
        self._value = 0
        self._maximum = 100
        self._streamlit_bar = streamlit_progress_bar

    def setValue(self, value):
        """设置进度值"""
        self._value = value
        if self._streamlit_bar is not None:
            # 计算百分比并更新Streamlit进度条
            progress = min(1.0, max(0.0, value / self._maximum))
            self._streamlit_bar.progress(progress)

    def setMaximum(self, maximum):
        """设置最大值"""
        self._maximum = maximum

    def value(self):
        """获取当前值"""
        return self._value

    def maximum(self):
        """获取最大值"""
        return self._maximum


def cellpose_segment(image: np.ndarray,
                     model_type: str = 'cyto2',
                     diameter: float = None,
                     channels: list = None,
                     progress_bar=None,
                     use_gpu: bool = False,
                     batch_size: int = 8,
                     normalize: dict = None) -> np.ndarray:
    """
    使用Cellpose进行细胞分割

    Args:
        image: 输入图像 (H, W) 或 (H, W, C)
        model_type: 模型类型 ('cyto', 'cyto2', 'nuclei')
        diameter: 细胞直径(像素),None为自动检测
        channels: 通道配置 [细胞质通道, 细胞核通道]
                 灰度图: [0, 0]
                 RGB: [2, 3] (绿色为细胞质,蓝色为细胞核)
        progress_bar: Streamlit进度条对象
        use_gpu: 是否使用GPU加速
        batch_size: 批处理大小,用于处理大图像(默认8)
        normalize: 归一化参数字典,例如 {"tile_norm_blocksize": 0}
                  None表示使用默认归一化

    Returns:
        分割掩码,每个细胞有唯一标签
    """
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose not installed")

    # 获取CPU核心数用于优化
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"Cellpose segmentation: model={model_type}, diameter={diameter}, CPU cores={cpu_count}")

    # 默认通道配置
    if channels is None:
        if len(image.shape) == 2:
            channels = [0, 0]  # 灰度图
        else:
            channels = [0, 0]  # RGB默认使用灰度

    # 加载模型 - 使用GPU参数优化
    model = models.CellposeModel(gpu=use_gpu, model_type=model_type)

    # 创建进度回调对象
    progress_callback = None
    if progress_bar is not None:
        progress_callback = StreamlitProgressBar(progress_bar)

    # 执行分割 (Cellpose 4.x返回3个值: masks, flows, styles)
    # 构建eval参数
    eval_params = {
        'diameter': diameter,
        'channels': channels,
        'flow_threshold': 0.4,
        'cellprob_threshold': 0.0,
        'progress': progress_callback,
        'batch_size': batch_size
    }

    # 添加normalize参数（如果提供）
    if normalize is not None:
        eval_params['normalize'] = normalize

    masks, flows, styles = model.eval(image, **eval_params)

    logger.debug(f"Cellpose detected {len(np.unique(masks))-1} cells")

    return masks
