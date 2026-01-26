"""
CellViT深度学习细胞分割模块

使用CellViT预训练模型进行细胞核分割
CellViT是基于Vision Transformer的细胞分割模型，主要用于病理切片图像

注意：
- CellViT通过subprocess在专门的环境中运行（env_cellvit）
- 主应用可以在任何环境中运行，CellViT会自动调用正确的环境
- 模型主要为大型病理切片设计，这里提供了适配小图像的接口
"""
import os
import sys
import subprocess
import pickle
import tempfile
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Optional, Tuple


def cellvit_segment(
    image: np.ndarray,
    model_type: str = "CellViT-256",
    use_gpu: bool = False,
    target_size: int = 256,
    progress_bar=None
) -> np.ndarray:
    """
    使用CellViT进行细胞核分割

    通过subprocess在cellvit环境中执行推理，避免环境冲突

    Args:
        image: 输入图像 (H, W, C) RGB格式或 (H, W) 灰度图
        model_type: 模型类型，目前支持 "CellViT-256"
        use_gpu: 是否使用GPU加速
        target_size: 目标图像大小（CellViT-256使用256x256）
        progress_bar: Streamlit进度条对象

    Returns:
        分割掩码，每个细胞核有唯一标签

    Raises:
        RuntimeError: 如果推理失败
    """
    # 检查图像格式
    original_shape = image.shape
    if image.ndim == 2:
        # 灰度图转RGB
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        # RGBA转RGB
        image = image[:, :, :3]
    elif image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    logger.info(f"CellViT segmentation: image_shape={original_shape}, target_size={target_size}")

    # 获取项目根目录和环境路径
    project_root = Path(__file__).parent.parent.parent.parent
    cellvit_env_path = project_root / "env_cellvit"
    worker_script = project_root / "src" / "core" / "segmentation" / "cellvit_worker.py"

    # 检查环境和脚本是否存在
    if not cellvit_env_path.exists():
        raise RuntimeError(
            f"CellViT environment not found at {cellvit_env_path}\n"
            "Please create the environment first."
        )

    if not worker_script.exists():
        raise RuntimeError(
            f"CellViT worker script not found at {worker_script}"
        )

    # 确定Python可执行文件路径
    if sys.platform == "win32":
        python_exe = cellvit_env_path / "python.exe"
        if not python_exe.exists():
            python_exe = cellvit_env_path / "Scripts" / "python.exe"
    else:
        python_exe = cellvit_env_path / "bin" / "python"

    if not python_exe.exists():
        raise RuntimeError(
            f"Python executable not found in CellViT environment at {python_exe}"
        )

    logger.info(f"Using CellViT environment: {cellvit_env_path}")
    logger.info(f"Python executable: {python_exe}")

    try:
        # 更新进度条
        if progress_bar is not None:
            progress_bar.progress(0.2)

        # 创建临时文件用于数据传递
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as input_file:
            input_path = input_file.name
            # 准备输入数据
            input_data = {
                'image': image,
                'model_type': model_type,
                'use_gpu': use_gpu,
                'target_size': target_size
            }
            pickle.dump(input_data, input_file)

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as output_file:
            output_path = output_file.name

        # 更新进度条
        if progress_bar is not None:
            progress_bar.progress(0.4)

        logger.info("Calling CellViT worker in subprocess...")

        # 调用worker脚本
        result = subprocess.run(
            [str(python_exe), str(worker_script), input_path, output_path],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )

        # 更新进度条
        if progress_bar is not None:
            progress_bar.progress(0.8)

        # 检查执行结果
        if result.returncode != 0:
            error_msg = f"CellViT worker failed with return code {result.returncode}\n"
            error_msg += f"STDOUT: {result.stdout}\n"
            error_msg += f"STDERR: {result.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # 打印CellViT worker的调试输出（成功时）
        if result.stdout:
            logger.info(f"CellViT worker stdout:\n{result.stdout}")
        if result.stderr:
            logger.info(f"CellViT worker stderr:\n{result.stderr}")

        # 读取输出结果
        with open(output_path, 'rb') as f:
            output_data = pickle.load(f)

        # 清理临时文件
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass

        # 更新进度条
        if progress_bar is not None:
            progress_bar.progress(1.0)

        # 检查结果
        if not output_data['success']:
            raise RuntimeError(f"CellViT inference failed: {output_data['error']}")

        mask = output_data['mask']
        num_cells = len(np.unique(mask)) - 1
        logger.info(f"CellViT detected {num_cells} cells")

        return mask

    except subprocess.TimeoutExpired:
        logger.error("CellViT worker timeout")
        raise RuntimeError("CellViT inference timeout (>5 minutes)")
    except Exception as e:
        logger.error(f"CellViT segmentation failed: {str(e)}")
        raise RuntimeError(f"CellViT segmentation error: {str(e)}")
