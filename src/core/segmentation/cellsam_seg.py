"""
CellSAM深度学习细胞分割模块

使用CellSAM (Segment Anything Model for Cells) 进行细胞分割
CellSAM是基于SAM的细胞分割模型，适用于多种细胞图像

注意：
- CellSAM直接在当前环境中运行（cellpose_gpu环境已包含所需依赖）
- 支持自动提示点生成进行细胞分割
- 需要预先下载SAM模型检查点文件
"""
import time
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Optional

# 延迟导入以避免启动时的开销
_sam_model_cache = {}


def cellsam_segment(
    image: np.ndarray,
    model_type: str = "vit_b",
    use_gpu: bool = False,
    points_per_side: int = 32,
    progress_bar=None
) -> np.ndarray:
    """
    使用CellSAM进行细胞分割

    直接在当前环境中执行推理

    Args:
        image: 输入图像 (H, W, C) RGB格式或 (H, W) 灰度图
        model_type: 模型类型，支持 "vit_b", "vit_l", "vit_h"
        use_gpu: 是否使用GPU加速
        points_per_side: 自动生成提示点的密度（每边的点数）
        progress_bar: Streamlit进度条对象

    Returns:
        分割掩码，每个细胞有唯一标签

    Raises:
        RuntimeError: 如果推理失败
    """
    start_time = time.time()

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

    logger.info(f"CellSAM segmentation: image_shape={original_shape}, model_type={model_type}")

    try:
        # 导入必要的库
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        # 更新进度条
        if progress_bar is not None:
            progress_bar.progress(0.1)

        # 设置设备
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # 加载模型（使用缓存）
        model_key = f"{model_type}_{device}"
        if model_key not in _sam_model_cache:
            logger.info(f"Loading SAM model: {model_type}")
            model_load_start = time.time()

            # 确定模型检查点路径
            project_root = Path(__file__).parent.parent.parent.parent
            checkpoint_dir = project_root / "models" / "sam"

            # 模型检查点文件名映射
            checkpoint_files = {
                "vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth"
            }

            if model_type not in checkpoint_files:
                raise ValueError(f"Unsupported model type: {model_type}")

            checkpoint_path = checkpoint_dir / checkpoint_files[model_type]

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Model checkpoint not found at {checkpoint_path}\n"
                    f"Please download the model from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
                )

            # 加载SAM模型
            sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
            sam.to(device=device)

            _sam_model_cache[model_key] = sam

            model_load_time = time.time() - model_load_start
            logger.info(f"Model loaded in {model_load_time:.2f}s")
        else:
            sam = _sam_model_cache[model_key]
            logger.info(f"Using cached model: {model_type}")

        # 更新进度条
        if progress_bar is not None:
            progress_bar.progress(0.3)

        # 创建自动掩码生成器
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # 最小区域面积，过滤小碎片
        )

        # 更新进度条
        if progress_bar is not None:
            progress_bar.progress(0.5)

        # 执行推理
        logger.info("Starting inference...")
        inference_start = time.time()

        # SAM需要RGB格式的uint8图像
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        masks = mask_generator.generate(image)

        inference_time = time.time() - inference_start
        logger.info(f"Inference completed in {inference_time:.2f}s, generated {len(masks)} masks")

        # 更新进度条
        if progress_bar is not None:
            progress_bar.progress(0.8)

        # 后处理：将多个掩码合并为实例分割图
        logger.info("Post-processing masks...")
        postprocess_start = time.time()

        h, w = image.shape[:2]
        image_area = h * w
        instance_map = np.zeros((h, w), dtype=np.int32)

        # 过滤掩码：移除过大或过小的掩码
        filtered_masks = []
        for mask_data in masks:
            mask = mask_data['segmentation']
            mask_area = np.sum(mask)
            area_ratio = mask_area / image_area

            # 过滤条件：
            # 1. 掩码面积不能超过图像面积的 50%（避免背景掩码）
            # 2. 掩码面积不能小于 100 像素（已由 SAM 的 min_mask_region_area 处理）
            if area_ratio < 0.5:
                filtered_masks.append(mask_data)
            else:
                logger.debug(f"Filtered out large mask: area={mask_area}, ratio={area_ratio:.2%}")

        logger.info(f"Filtered masks: {len(masks)} -> {len(filtered_masks)}")

        # 按照预测质量排序（从高到低）
        filtered_masks = sorted(filtered_masks, key=lambda x: x['predicted_iou'], reverse=True)

        # 逐个添加掩码，避免重叠
        for idx, mask_data in enumerate(filtered_masks, start=1):
            mask = mask_data['segmentation']
            # 只在未被标记的区域添加新掩码
            instance_map[mask & (instance_map == 0)] = idx

        postprocess_time = time.time() - postprocess_start
        logger.info(f"Post-processing completed in {postprocess_time:.2f}s")

        # 统计检测到的细胞数量
        num_cells = len(np.unique(instance_map)) - 1
        total_time = time.time() - start_time
        logger.info(f"CellSAM detected {num_cells} cells in {total_time:.2f}s")

        # 更新进度条
        if progress_bar is not None:
            progress_bar.progress(1.0)

        return instance_map

    except Exception as e:
        logger.error(f"CellSAM segmentation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"CellSAM segmentation error: {str(e)}")
