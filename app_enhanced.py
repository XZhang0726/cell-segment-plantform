"""
细胞分割平台 - 增强版 Streamlit UI

新增功能：
1. 处理更多类型的细胞图像（预处理选项）
2. 批量处理多张图像
3. 可视化对比工具
4. 批量导出功能
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import time
import pandas as pd
import zipfile
import io
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from skimage import measure
import plotly.express as px

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.segmentation import CellSegmenter, SegmentationMethod
from src.core.features import extract_cell_features, get_feature_statistics, extract_advanced_cell_features

# 导入ML模块
from src.ml.clustering import perform_kmeans, perform_dbscan, perform_hierarchical, perform_gmm, find_optimal_clusters
from src.ml.dimensionality_reduction import apply_pca, apply_tsne, apply_umap
from src.ml.feature_analysis import analyze_feature_importance
from src.ml.anomaly_detection import (detect_isolation_forest, detect_lof,
                                       detect_one_class_svm, detect_elliptic_envelope,
                                       get_anomaly_statistics,
                                       visualize_isolation_forest, visualize_lof,
                                       visualize_one_class_svm, visualize_elliptic_envelope)

# 导入新增ML模块
from src.ml.supervised_learning import (
    train_supervised_model, compare_models_automl,
    evaluate_classification, evaluate_regression,
    save_model, load_model,
    plot_feature_importance, plot_confusion_matrix, plot_roc_curves,
    plot_prediction_vs_actual, plot_residuals, plot_learning_curves
)
from src.ml.active_learning import (
    active_learning_workflow, uncertainty_sampling, query_by_committee,
    bayesian_optimization_loop,
    plot_uncertainty_intervals, plot_acquisition_function,
    plot_optimization_trajectory, plot_convergence
)
from src.ml.virtual_screening import (
    screen_dataset, batch_screen_files,
    rank_by_prediction, filter_by_confidence, select_top_candidates,
    plot_prediction_distribution, plot_confidence_distribution,
    plot_top_candidates, plot_prediction_vs_confidence
)

# 导入i18n翻译模块
from locales.i18n import t, get_i18n

def build_help_markdown():
    """构建帮助文档的markdown内容"""
    i18n = get_i18n()
    help_data = i18n.translations.get('help', {})

    md = f"### {help_data.get('overview', {}).get('title', '')}\n\n"
    md += f"{help_data.get('overview', {}).get('content', '')}\n\n"

    # 添加各个标签页的说明
    for i in range(1, 11):
        tab = help_data.get(f'tab{i}', {})
        if tab:
            md += f"#### {tab.get('title', '')}\n"
            md += f"{tab.get('description', '')}\n"
            features = tab.get('features', [])
            if features:
                for feat in features:
                    md += f"- {feat}\n"
            usage = tab.get('usage', '')
            if usage:
                md += f"\n**{t('common.usage') if 'common.usage' in dir() else 'Usage'}**: {usage}\n"
            md += "\n"

    return md

# 检查GPU可用性
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        # 检查GPU兼容性（RTX 5070需要PyTorch 2.10.0+cu128支持）
        if "RTX 5070" in GPU_NAME or "RTX 50" in GPU_NAME:
            # 检查PyTorch版本是否支持RTX 50系列
            pytorch_version = torch.__version__
            if "cu128" in pytorch_version or (hasattr(torch.version, 'cuda') and torch.version.cuda == "12.8"):
                GPU_COMPATIBLE = True
                GPU_WARNING = None
            else:
                GPU_COMPATIBLE = False
                GPU_WARNING = t('messages.gpu_incompatible', gpu_name=GPU_NAME, version=pytorch_version)
        else:
            GPU_COMPATIBLE = True
            GPU_WARNING = None
    else:
        GPU_NAME = None
        GPU_COMPATIBLE = False
        GPU_WARNING = None
except:
    GPU_AVAILABLE = False
    GPU_NAME = None
    GPU_COMPATIBLE = False
    GPU_WARNING = None

# 检查CellViT环境是否存在
def check_cellvit_environment():
    """检查CellViT专用环境是否存在"""
    try:
        from pathlib import Path
        project_root = Path(__file__).parent
        cellvit_env_path = project_root / "env_cellvit"

        if cellvit_env_path.exists():
            return True, "env_cellvit"
        else:
            return False, "not_found"
    except:
        return False, "unknown"

CELLVIT_ENV_OK, CELLVIT_ENV_STATUS = check_cellvit_environment()

# 页面配置
st.set_page_config(
    page_title="Cell Segmentation Platform - Enhanced",
    page_icon="🔬",
    layout="wide"
)

# 初始化session state
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = {}


def preprocess_image(image, denoise=False, enhance=False, normalize=False):
    """
    图像预处理

    Args:
        image: 输入图像
        denoise: 是否去噪
        enhance: 是否增强对比度
        normalize: 是否归一化

    Returns:
        预处理后的图像
    """
    processed = image.copy()

    if denoise:
        # 高斯去噪
        processed = cv2.GaussianBlur(processed, (5, 5), 0)

    if enhance:
        # CLAHE对比度增强
        if len(processed.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        else:
            # 转换到LAB空间进行增强
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)

    if normalize:
        # 归一化到0-255
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)

    return processed


def extract_individual_cells(image_np, mask, min_area=100):
    """
    从分割掩码中提取单个细胞样本

    使用标签值作为ID，确保与形态学特征的cell_id一致

    Args:
        image_np: 原始图像
        mask: 分割掩码（标签掩码，每个细胞有唯一标签值）
        min_area: 最小细胞面积阈值

    Returns:
        individual_cells: 单个细胞图像列表
        cell_info: 细胞信息列表（包含位置、面积等）
    """
    from loguru import logger

    individual_cells = []
    cell_info = []

    # 获取所有唯一标签（排除背景0）
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]

    # 调试信息
    logger.info(f"[Cell Extraction] 掩码中的唯一标签数: {len(unique_labels)}")
    logger.info(f"[Cell Extraction] 掩码形状: {mask.shape}, 图像形状: {image_np.shape}")
    logger.info(f"[Cell Extraction] 掩码值范围: {mask.min()} - {mask.max()}")

    # 检查是否为二值掩码，如果是则转换为实例分割掩码
    if len(unique_labels) == 1:
        logger.info(f"[Cell Extraction] 检测到二值掩码，应用连通组件标记...")
        # 将掩码转换为二值图（0和1）
        binary_mask = (mask > 0).astype(np.uint8)

        # 应用连通组件标记
        from scipy.ndimage import label as scipy_label
        labeled_mask, num_features = scipy_label(binary_mask)

        logger.info(f"[Cell Extraction] 连通组件标记完成，检测到 {num_features} 个区域")

        # 更新mask和unique_labels
        mask = labeled_mask
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels > 0]
        logger.info(f"[Cell Extraction] 更新后的唯一标签数: {len(unique_labels)}")

    # 计算图像总面积，用于过滤异常大的区域
    total_area = image_np.shape[0] * image_np.shape[1]
    max_area = total_area * 0.5  # 最大面积为图像总面积的50%

    for label in unique_labels:
        # 创建当前细胞的二值掩码
        cell_mask_binary = (mask == label).astype(np.uint8)

        # 计算面积
        area = np.sum(cell_mask_binary)

        # 过滤太小的区域
        if area < min_area:
            continue

        # 过滤异常大的区域（可能是背景噪声或多个连接的细胞）
        if area > max_area:
            logger.warning(f"[Cell Extraction] 跳过异常大的区域 (label={int(label)}, area={area}, 占比={area/total_area*100:.1f}%)")
            continue

        # 获取边界框
        coords = np.where(cell_mask_binary > 0)
        if len(coords[0]) == 0:
            continue

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # 添加边距
        padding = 5
        y1 = max(0, y_min - padding)
        x1 = max(0, x_min - padding)
        y2 = min(image_np.shape[0], y_max + padding + 1)
        x2 = min(image_np.shape[1], x_max + padding + 1)

        # 裁剪细胞图像和掩码
        cell_image = image_np[y1:y2, x1:x2].copy()
        cell_mask = cell_mask_binary[y1:y2, x1:x2] * 255

        # 调试信息：显示前3个细胞的详细信息
        if len(individual_cells) < 3:
            logger.info(f"[Cell Extraction] 细胞 {int(label)}: bbox=({x1},{y1},{x2},{y2}), "
                       f"提取图像大小={cell_image.shape}, 面积={area}")

        individual_cells.append({
            'image': cell_image,
            'mask': cell_mask,
            'bbox': (x1, y1, x2, y2),
            'label': int(label)  # 使用标签值作为ID
        })

        cell_info.append({
            'id': int(label),  # 使用标签值作为ID，与形态学特征的cell_id一致
            'area': int(area),
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'center': (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
        })

    return individual_cells, cell_info


def process_single_image_worker(args):
    """
    并行处理单张图像的工作函数

    Args:
        args: 包含(image_data, filename, method, params, preprocess_options, postprocess_options)的元组

    Returns:
        处理结果字典
    """
    image_data, filename, method, params, preprocess_options, postprocess_options = args

    try:
        # 将图像数据转换为numpy数组
        image_np = np.array(image_data)

        # 处理图像
        result = segment_single_image(
            image_np,
            method,
            params,
            preprocess_options,
            postprocess_options
        )

        return {
            'filename': filename,
            'result': result,
            'image': image_np,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'filename': filename,
            'result': None,
            'image': None,
            'success': False,
            'error': str(e)
        }


def colorize_instance_mask(mask):
    """
    为实例分割掩码着色，每个细胞使用不同颜色

    Args:
        mask: 实例分割掩码，每个细胞有唯一标签

    Returns:
        彩色掩码 (H, W, 3)
    """
    # 获取所有唯一标签（排除背景0）
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]

    # 创建彩色掩码
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 为每个细胞分配随机颜色
    np.random.seed(42)  # 固定随机种子以保持一致性
    for label in unique_labels:
        color = np.random.randint(0, 255, 3)
        colored_mask[mask == label] = color

    return colored_mask


def segment_single_image(image_np, method, params, preprocess_options, postprocess_options=None):
    """
    分割单张图像

    Args:
        image_np: 输入图像
        method: 分割方法
        params: 方法参数
        preprocess_options: 预处理选项
        postprocess_options: 后处理选项（包括区域闭合等）

    Returns:
        分割结果字典
    """
    if postprocess_options is None:
        postprocess_options = {}

    # 预处理
    processed_image = preprocess_image(
        image_np,
        denoise=preprocess_options.get('denoise', False),
        enhance=preprocess_options.get('enhance', False),
        normalize=preprocess_options.get('normalize', False)
    )

    # 创建分割器
    method_map = {
        t('methods.otsu'): SegmentationMethod.OTSU,
        t('methods.adaptive'): SegmentationMethod.ADAPTIVE,
        t('methods.watershed'): SegmentationMethod.WATERSHED,
        t('methods.canny'): SegmentationMethod.EDGE_CANNY,
        t('methods.cellpose'): SegmentationMethod.CELLPOSE,
        t('methods.cellvit'): SegmentationMethod.CELLVIT,
        t('methods.cellsam'): SegmentationMethod.CELLSAM
    }

    seg_method = method_map[method]
    segmenter = CellSegmenter(method=seg_method)

    # 执行分割
    start_time = time.time()
    mask = segmenter.segment(processed_image, **params)
    elapsed_time = time.time() - start_time

    # 后处理：形态学闭运算
    if postprocess_options.get('closing', False):
        kernel_size = postprocess_options.get('closing_kernel_size', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 检查是否为标签掩码（包含多个不同的标签值）
        unique_labels = np.unique(mask)
        is_labeled_mask = len(unique_labels) > 2  # 超过2个值说明是标签掩码（不只是0和1或0和255）

        if is_labeled_mask:
            # 对于实例分割掩码（CellViT、CellSAM、Cellpose等），跳过形态学闭运算
            # 因为它会破坏实例标签，导致细胞边界框错误
            from loguru import logger
            logger.warning("[后处理] 检测到实例分割掩码，跳过形态学闭运算以保护标签完整性")
            pass
        else:
            # 对于二值掩码（Otsu、自适应阈值等），直接进行闭运算
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 提取单个细胞（如果需要）
    individual_cells = None
    cell_info = None
    if postprocess_options.get('extract_cells', False):
        min_area = postprocess_options.get('min_cell_area', 100)
        individual_cells, cell_info = extract_individual_cells(image_np, mask, min_area)

    # 检查是否为实例分割掩码（有多个唯一标签）
    unique_labels = np.unique(mask)
    num_instances = len(unique_labels[unique_labels > 0])
    is_instance_segmentation = num_instances > 1

    # 归一化掩码或着色
    if is_instance_segmentation:
        # 实例分割：为每个细胞着色
        mask_display = colorize_instance_mask(mask)
    else:
        # 二值分割：归一化为灰度图
        if mask.max() > 0:
            mask_display = (mask / mask.max() * 255).astype(np.uint8)
        else:
            mask_display = mask.astype(np.uint8)

    # 创建叠加图
    if len(processed_image.shape) == 2:
        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = processed_image.copy()

    if is_instance_segmentation:
        # 实例分割：使用彩色掩码叠加
        colored_mask = colorize_instance_mask(mask)
        overlay_mask = colored_mask.copy()
        result = cv2.addWeighted(image_rgb, 0.7, overlay_mask, 0.3, 0)
    else:
        # 二值分割：使用红色叠加
        overlay = image_rgb.copy()
        overlay[mask > 0] = [255, 0, 0]
        result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

    # 统计信息
    foreground_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    foreground_ratio = foreground_pixels / total_pixels * 100

    # 计算检测到的细胞区域数量（应用min_area过滤以保持一致性）
    if postprocess_options.get('extract_cells', False) or postprocess_options.get('extract_morphology', False):
        # 如果启用了细胞提取或形态学特征提取，应用min_area过滤
        min_area = postprocess_options.get('min_cell_area', 100)
        regions = measure.regionprops(mask)
        num_regions = sum(1 for region in regions if region.area >= min_area)
    else:
        # 否则显示所有检测到的细胞
        unique_labels = np.unique(mask)
        num_regions = len(unique_labels[unique_labels > 0]) if len(unique_labels) > 1 else 0

    return {
        'mask': mask_display,
        'labeled_mask': mask,
        'overlay': result,
        'processed_image': processed_image,
        'foreground_pixels': foreground_pixels,
        'foreground_ratio': foreground_ratio,
        'processing_time': elapsed_time,
        'num_regions': num_regions,
        'individual_cells': individual_cells,
        'cell_info': cell_info
    }


def create_comparison_view(image_np, methods, params_dict, preprocess_options, postprocess_options=None):
    """
    创建多方法对比视图

    Args:
        image_np: 输入图像
        methods: 方法列表
        params_dict: 参数字典
        preprocess_options: 预处理选项
        postprocess_options: 后处理选项

    Returns:
        对比结果字典
    """
    results = {}

    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, method in enumerate(methods):
        # 更新进度提示
        progress = (idx + 1) / len(methods)
        progress_bar.progress(progress)

        if method == "Cellpose深度学习":
            status_text.text(f"🧠 正在处理 {idx+1}/{len(methods)}: {method}（深度学习模型，请稍候）...")
        else:
            status_text.text(f"⚙️ 正在处理 {idx+1}/{len(methods)}: {method}...")

        params = params_dict.get(method, {})
        result = segment_single_image(image_np, method, params, preprocess_options, postprocess_options)
        results[method] = result

    # 清除进度提示
    progress_bar.empty()
    status_text.empty()

    return results


# 标题
col_title, col_help = st.columns([6, 1])
with col_title:
    st.title(t('app.title_enhanced'))
    st.markdown(t('app.subtitle_enhanced'))

    # GPU状态指示器
    if GPU_AVAILABLE and GPU_COMPATIBLE:
        st.success(t('messages.gpu_available', gpu_name=GPU_NAME))
    elif GPU_AVAILABLE and not GPU_COMPATIBLE:
        st.error(t('messages.gpu_detected_incompatible', gpu_name=GPU_NAME))
        st.warning(GPU_WARNING)
    else:
        st.info(t('messages.gpu_unavailable'))
with col_help:
    st.write("")  # 添加空行对齐
    with st.popover(t('app.help_title')):
        st.markdown(build_help_markdown())

# ==================== 模型融合流程函数 ====================
def run_fusion_pipeline(image, selected_models, strategy, iou_threshold, min_vote_count, weights, model_params, display_col,
                       model_reliabilities=None, conflict_threshold=0.6, postprocess_options=None):
    """执行完整的融合流程（支持简单策略和DST高级融合）"""
    # 使用优化版本的实例匹配（57倍加速）
    from src.core.fusion import match_instances, fuse_instances, fuse_instances_dst, generate_confidence_maps
    from src.core.fusion.uncertainty import compute_disagreement_map, compute_model_consistency
    from skimage.color import label2rgb
    import matplotlib.pyplot as plt

    # 模型名称映射
    model_name_mapping = {
        "cellpose": t('methods.cellpose'),
        "cellvit": t('methods.cellvit'),
        "cellsam": t('methods.cellsam'),
        "watershed": t('methods.watershed'),
        "otsu": t('methods.otsu'),
        "adaptive": t('methods.adaptive'),
        "canny": t('methods.canny')
    }

    with display_col:
        with st.spinner(t('messages.running_multi_model_inference')):
            # 1. 运行所有选择的模型
            masks_list = []
            model_names = []

            progress_bar = st.progress(0)

            # 默认预处理和后处理选项
            preprocess_options = {
                'denoise': False,
                'enhance': False,
                'normalize': False
            }

            # 使用传入的后处理选项，如果没有则使用默认值
            if postprocess_options is None:
                postprocess_options = {
                    'closing': True,
                    'closing_kernel_size': 5,
                    'extract_cells': False,
                    'min_cell_area': 100,
                    'extract_morphology': False,
                    'use_advanced_features': False
                }
            else:
                # 确保所有必需的键都存在
                default_postprocess = {
                    'closing': True,
                    'closing_kernel_size': 5,
                    'extract_cells': False,
                    'min_cell_area': 100,
                    'extract_morphology': False,
                    'use_advanced_features': False
                }
                default_postprocess.update(postprocess_options)
                postprocess_options = default_postprocess

            for idx, model_name in enumerate(selected_models):
                st.text(t('messages.running_model', model_name=model_name))

                try:
                    # 获取完整的方法名称
                    method = model_name_mapping[model_name]

                    # 准备参数
                    params = {}
                    if model_name == "cellpose":
                        params['model_type'] = 'cyto2'
                        params['diameter'] = model_params.get('cellpose_diameter', 30)
                        params['use_gpu'] = model_params.get('cellpose_gpu', True)
                    elif model_name == "cellvit":
                        params['model_type'] = 'CellViT-SAM-H'
                        params['target_size'] = model_params.get('cellvit_size', 256)
                        params['use_gpu'] = model_params.get('cellvit_gpu', True)
                    elif model_name == "cellsam":
                        params['model_type'] = 'vit_h'
                        params['points_per_side'] = model_params.get('cellsam_points', 32)
                        params['use_gpu'] = model_params.get('cellsam_gpu', True)
                    elif model_name == "watershed":
                        params['min_distance'] = model_params.get('watershed_min_distance', 10)
                        params['threshold_rel'] = model_params.get('watershed_threshold', 0.5)
                    elif model_name == "otsu":
                        # Otsu方法无需额外参数
                        pass
                    elif model_name == "adaptive":
                        params['block_size'] = model_params.get('adaptive_block_size', 21)
                        params['C'] = model_params.get('adaptive_c', 5)
                    elif model_name == "canny":
                        params['threshold1'] = model_params.get('canny_threshold1', 50)
                        params['threshold2'] = model_params.get('canny_threshold2', 150)

                    # 调用分割函数
                    result = segment_single_image(image, method, params, preprocess_options, postprocess_options)
                    masks_list.append(result['labeled_mask'])
                    model_names.append(model_name)

                except Exception as e:
                    st.error(t('messages.model_inference_failed', model_name=model_name, error=str(e)))
                    import traceback
                    st.text(traceback.format_exc())
                    continue

                progress_bar.progress((idx + 1) / len(selected_models))

            if len(masks_list) < 2:
                st.error(t('messages.at_least_two_models_needed'))
                return

            st.success(t('messages.model_inference_completed', count=len(masks_list)))

        # 创建融合进度条
        st.subheader(t('common.fusion_progress'))
        fusion_progress = st.progress(0)
        fusion_status = st.empty()

        # 步骤1: 实例匹配 (0% -> 33%)
        fusion_status.text(t('messages.fusion_step1_matching'))
        try:
            matched_groups = match_instances(masks_list, iou_threshold)
            fusion_progress.progress(0.33)
            fusion_status.text(t('messages.fusion_step1_completed', count=len(matched_groups)))
        except Exception as e:
            st.error(t('messages.instance_matching_failed', error=str(e)))
            return

        # 步骤2: 融合掩码 (33% -> 66%)
        fusion_status.text(t('messages.fusion_step2_fusing'))
        try:
            # 判断使用简单策略还是DST高级融合
            if strategy == 'dempster_shafer':
                # DST高级融合
                fusion_status.text(t('messages.fusion_step2_dst'))

                # 生成合成置信度图（基于距离变换，边界置信度低，中心置信度高）
                confidences_list = generate_confidence_maps(masks_list, model_names, model_reliabilities)

                fused_mask, dst_stats = fuse_instances_dst(
                    matched_groups, masks_list, confidences_list,
                    model_names, model_reliabilities,
                    image=image,
                    min_vote_count=min_vote_count,
                    conflict_threshold=conflict_threshold,
                    enable_watershed=True
                )

                fusion_progress.progress(0.66)

                # 显示策略使用统计
                strategy_counts = dst_stats.get('strategy_counts', {})
                dominant_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else "UNKNOWN"

                fusion_status.text(
                    t('messages.fusion_step2_dst_completed', count=np.max(fused_mask)) +
                    f" (平均冲突={dst_stats['average_conflict']:.3f}, 主要策略={dominant_strategy})"
                )
            else:
                # 简单融合策略
                weight_list = None
                if weights is not None:
                    weight_list = [weights.get(name, 1.0) for name in model_names]

                fused_mask = fuse_instances(
                    matched_groups, masks_list, strategy,
                    weights=weight_list, min_vote_count=min_vote_count
                )
                dst_stats = None

                fusion_progress.progress(0.66)
                fusion_status.text(t('messages.fusion_step2_completed', count=np.max(fused_mask)))
        except Exception as e:
            st.error(t('messages.fusion_failed', error=str(e)))
            import traceback
            st.text(traceback.format_exc())
            return

        # 步骤3: 计算不确定性 (66% -> 100%)
        fusion_status.text(t('messages.fusion_step3_calculating'))
        try:
            disagreement_map, consistency_score = compute_disagreement_map(masks_list)
            consistency_matrix, avg_consistency = compute_model_consistency(masks_list)
            fusion_progress.progress(1.0)
            fusion_status.text(t('messages.fusion_step3_completed', consistency=consistency_score))
        except Exception as e:
            st.error(t('messages.uncertainty_calculation_failed', error=str(e)))
            return

        # 完成提示
        st.success(t('messages.fusion_completed'))

        # 清除进度条和状态文本（可选，如果想保留就注释掉）
        # fusion_progress.empty()
        # fusion_status.empty()

        # 4.5. 提取单个细胞样本和形态学特征（如果需要）
        individual_cells = None
        cell_info = None
        morphology_features = None

        if postprocess_options.get('extract_cells', False) or postprocess_options.get('extract_morphology', False):
            min_area = postprocess_options.get('min_cell_area', 100)

            # 提取单个细胞样本
            if postprocess_options.get('extract_cells', False):
                try:
                    individual_cells, cell_info = extract_individual_cells(image, fused_mask, min_area)
                    st.info(f"已提取 {len(individual_cells)} 个单细胞样本")
                except Exception as e:
                    st.warning(f"单细胞提取失败: {str(e)}")

            # 提取形态学特征
            if postprocess_options.get('extract_morphology', False):
                try:
                    use_advanced = postprocess_options.get('use_advanced_features', False)
                    if use_advanced:
                        morphology_features = extract_advanced_cell_features(
                            fused_mask, image, min_area=min_area,
                            include_hu_moments=True,
                            include_intensity=True,
                            include_texture=True,
                            include_boundary=True,
                            include_advanced_shape=True
                        )
                    else:
                        morphology_features = extract_cell_features(image, fused_mask, min_area=min_area)

                    st.info(f"已提取 {len(morphology_features)} 个细胞的形态学特征")
                except Exception as e:
                    st.warning(f"形态学特征提取失败: {str(e)}")

        # 5. 显示结果
        st.subheader(t('common.fusion_results'))

        # 根据是否使用DST和单细胞提取决定显示哪些tab
        tab_names = [t('fusion.fused_mask_tab'), t('fusion.uncertainty_heatmap_tab'), t('fusion.model_comparison_tab')]

        # 如果启用了单细胞提取或形态学特征提取，添加单细胞分析tab
        has_cell_analysis = (individual_cells is not None and len(individual_cells) > 0) or (morphology_features is not None and len(morphology_features) > 0)
        if has_cell_analysis:
            tab_names.append("单细胞分析")

        # 如果使用了DST，添加DST分析tab
        if dst_stats is not None:
            tab_names.append(t('fusion.dst_analysis_tab'))

        result_tabs = st.tabs(tab_names)

        with result_tabs[0]:
            # 显示融合掩码
            try:
                fused_display = label2rgb(fused_mask, bg_label=0)
                st.image(fused_display, caption=t('fusion.fused_result_caption'), use_container_width=True)

                # 添加下载按钮
                from io import BytesIO
                fused_img = Image.fromarray((fused_display * 255).astype(np.uint8))
                buf = BytesIO()
                fused_img.save(buf, format='PNG')
                st.download_button(
                    label=t('fusion.download_fused_mask'),
                    data=buf.getvalue(),
                    file_name="fused_mask.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(t('messages.display_fusion_mask_failed', error=str(e)))

        with result_tabs[1]:
            # 显示不确定性热图（叠加在原图上）
            try:
                # 使用log处理实现平滑的颜色渐变
                disagreement_log = np.log1p(disagreement_map * 10) / np.log1p(10)

                # 准备原始图像（转换为RGB）
                if len(image.shape) == 2:
                    # 灰度图转RGB
                    base_image = np.stack([image] * 3, axis=-1)
                elif image.shape[2] == 4:
                    # RGBA转RGB
                    base_image = image[:, :, :3]
                else:
                    base_image = image.copy()

                # 归一化原始图像到0-1范围
                base_image = base_image.astype(np.float32)
                if base_image.max() > 1:
                    base_image = base_image / 255.0

                # 将不确定性热图转换为彩色图像（使用jet色图：蓝色=低不确定性，红色=高不确定性）
                from matplotlib import cm
                colormap = cm.get_cmap('jet')
                heatmap_colored = colormap(disagreement_log)[:, :, :3]  # 去掉alpha通道

                # Alpha混合：热图叠加到原图上（alpha=0.5表示50%透明度）
                alpha = 0.5
                overlay = base_image * (1 - alpha) + heatmap_colored * alpha

                # 保存分水岭细化区域信息（用于后续绘制轮廓）
                watershed_refined_mask = None
                if dst_stats is not None and 'watershed_refinement' in dst_stats:
                    watershed_info = dst_stats['watershed_refinement']
                    if watershed_info.get('refined', False) and 'refined_mask' in watershed_info:
                        watershed_refined_mask = watershed_info['refined_mask']

                # 根据图像尺寸自适应调整figure大小
                h, w = disagreement_map.shape
                aspect_ratio = w / h
                fig_height = 8
                fig_width = fig_height * aspect_ratio

                fig, ax = plt.subplots(figsize=(fig_width, fig_height))

                # 显示叠加后的图像
                ax.imshow(overlay, aspect='auto')

                # 绘制分水岭细化区域的虚线轮廓
                if watershed_refined_mask is not None:
                    from skimage import measure
                    # 找到refined_mask的轮廓
                    contours = measure.find_contours(watershed_refined_mask.astype(float), 0.5)
                    # 绘制虚线轮廓
                    for contour in contours:
                        ax.plot(contour[:, 1], contour[:, 0],
                               linestyle='--', linewidth=2, color='lime',
                               alpha=0.8, label='Watershed Refined' if contour is contours[0] else '')

                # 添加标题（包含分水岭细化信息）
                title = f'Uncertainty Heatmap Overlay (Consistency: {consistency_score:.2%})'
                if dst_stats is not None and 'watershed_refinement' in dst_stats:
                    watershed_info = dst_stats['watershed_refinement']
                    if watershed_info.get('refined', False):
                        refined_pct = watershed_info.get('refined_percentage', 0)
                        title += f'\n-- Watershed Refined: {refined_pct:.2f}% pixels (dashed contours)'
                ax.set_title(title, fontsize=14, pad=15, fontweight='bold')

                # 添加图例（如果有分水岭细化）
                if watershed_refined_mask is not None:
                    ax.legend(loc='upper right', fontsize=10, framealpha=0.8)

                ax.axis('off')

                # 添加colorbar（使用ScalarMappable创建）
                from matplotlib import cm
                from matplotlib.colors import Normalize
                norm = Normalize(vmin=0, vmax=1)
                sm = cm.ScalarMappable(cmap='jet', norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
                cbar.set_label('Uncertainty Level', rotation=270, labelpad=20, fontsize=11)
                cbar.ax.tick_params(labelsize=9)

                # 调整布局，避免colorbar被裁剪
                plt.tight_layout()

                st.pyplot(fig)

                # 添加下载按钮
                from io import BytesIO
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label=t('fusion.download_disagreement_heatmap'),
                    data=buf.getvalue(),
                    file_name="uncertainty_heatmap.png",
                    mime="image/png"
                )

                plt.close(fig)
            except Exception as e:
                st.error(t('messages.display_uncertainty_heatmap_failed', error=str(e)))

        with result_tabs[2]:
            # 模型对比
            try:
                cols = st.columns(len(model_names))
                for idx, (model_name, mask) in enumerate(zip(model_names, masks_list)):
                    with cols[idx]:
                        display = label2rgb(mask, bg_label=0)
                        st.image(display, caption=f"{model_name}\n({np.max(mask)} cells)", use_container_width=True)

                        # 添加下载按钮
                        from io import BytesIO
                        model_img = Image.fromarray((display * 255).astype(np.uint8))
                        buf = BytesIO()
                        model_img.save(buf, format='PNG')
                        st.download_button(
                            label=t('common.download'),
                            data=buf.getvalue(),
                            file_name=f"{model_name}_result.png",
                            mime="image/png",
                            key=f"download_{model_name}_{idx}"
                        )
            except Exception as e:
                st.error(t('messages.display_model_comparison_failed', error=str(e)))

        # 单细胞分析tab（仅在启用单细胞提取或形态学特征提取时显示）
        if has_cell_analysis:
            # 计算单细胞分析tab的索引（总是在第4个位置，索引为3）
            cell_tab_idx = 3
            with result_tabs[cell_tab_idx]:
                st.markdown("### 单细胞分析")

                # 显示单细胞样本
                if individual_cells is not None and len(individual_cells) > 0:
                    st.markdown("#### 单细胞样本")
                    st.caption(f"共提取 {len(individual_cells)} 个单细胞样本（面积 ≥ {postprocess_options.get('min_cell_area', 100)} 像素）")

                    # 使用列布局显示单细胞样本（每行4个）
                    num_cells = len(individual_cells)
                    cells_per_row = 4
                    num_rows = (num_cells + cells_per_row - 1) // cells_per_row

                    for row_idx in range(min(num_rows, 5)):  # 最多显示5行（20个细胞）
                        cols = st.columns(cells_per_row)
                        for col_idx in range(cells_per_row):
                            cell_idx = row_idx * cells_per_row + col_idx
                            if cell_idx < num_cells:
                                with cols[col_idx]:
                                    cell_img = individual_cells[cell_idx]['image']
                                    info = cell_info[cell_idx]
                                    st.image(cell_img, caption=f"细胞 {info['id']}", use_container_width=True)
                                    st.caption(f"面积: {info['area']} px")

                    if num_cells > 20:
                        st.info(f"仅显示前20个细胞样本，共有 {num_cells} 个细胞")

                # 显示形态学特征
                if morphology_features is not None and len(morphology_features) > 0:
                    st.markdown("#### 形态学特征")
                    st.caption(f"共提取 {len(morphology_features)} 个细胞的形态学特征")

                    # 转换为DataFrame并显示
                    import pandas as pd
                    features_df = pd.DataFrame(morphology_features)

                    # 显示统计摘要
                    st.markdown("**特征统计摘要**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("细胞数量", len(features_df))
                    with col2:
                        st.metric("平均面积", f"{features_df['area_pixels'].mean():.1f} px")
                    with col3:
                        st.metric("平均周长", f"{features_df['perimeter_pixels'].mean():.1f} px")
                    with col4:
                        if 'circularity' in features_df.columns:
                            st.metric("平均圆度", f"{features_df['circularity'].mean():.3f}")

                    # 显示特征表格
                    st.markdown("**详细特征表**")
                    st.dataframe(features_df, use_container_width=True, height=300)

                    # 下载按钮
                    csv = features_df.to_csv(index=False)
                    st.download_button(
                        label="下载形态学特征 (CSV)",
                        data=csv,
                        file_name=f"fusion_morphology_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

        # DST分析tab（仅在使用DST融合时显示）
        if dst_stats is not None:
            # 计算DST分析tab的索引（如果有单细胞分析tab，则在第5个位置，索引为4；否则在第4个位置，索引为3）
            dst_tab_idx = 4 if has_cell_analysis else 3
            with result_tabs[dst_tab_idx]:
                st.markdown(t('fusion.dst_analysis_title'))

                # DST统计摘要
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(t('metrics.fused_instances'), dst_stats['fused_count'])
                with col2:
                    st.metric(t('metrics.average_conflict'), f"{dst_stats['average_conflict']:.3f}")
                with col3:
                    st.metric(t('metrics.average_uncertainty'), f"{dst_stats['average_uncertainty']:.3f}")
                with col4:
                    st.metric(t('metrics.high_conflict_instances'), dst_stats['high_conflict_count'])

                # 策略分布统计
                st.markdown(t('fusion.adaptive_strategy_distribution_title'))
                st.caption(t('fusion.adaptive_strategy_description'))

                # 显示置信度和冲突度分布
                if 'confidence_distribution' in dst_stats and 'conflict_distribution' in dst_stats:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(t('fusion.confidence_distribution'))
                        conf_dist = dst_stats['confidence_distribution']
                        st.text(f"{t('metrics.range')}: [{conf_dist['min']:.3f}, {conf_dist['max']:.3f}]")
                        st.text(f"{t('metrics.mean')}: {conf_dist['mean']:.3f}")
                        st.text(f"{t('metrics.std')}: {conf_dist['std']:.3f}")

                        # 判断置信度变化是否足够
                        if conf_dist['std'] < 0.05:
                            st.warning(t('messages.confidence_change_small'))

                    with col2:
                        st.markdown(t('fusion.conflict_distribution'))
                        conflict_dist = dst_stats['conflict_distribution']
                        st.text(f"{t('metrics.range')}: [{conflict_dist['min']:.3f}, {conflict_dist['max']:.3f}]")
                        st.text(f"{t('metrics.mean')}: {conflict_dist['mean']:.3f}")
                        st.text(f"{t('metrics.std')}: {conflict_dist['std']:.3f}")

                        # 判断冲突度变化是否足够
                        if conflict_dist['std'] < 0.05:
                            st.warning(t('messages.conflict_change_small'))

                    st.markdown("---")

                strategy_counts = dst_stats.get('strategy_counts', {})
                if strategy_counts and sum(strategy_counts.values()) > 0:
                    # 创建策略分布可视化
                    import pandas as pd

                    # 准备数据
                    strategy_data = []
                    total_count = sum(strategy_counts.values())
                    for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
                        if count > 0:
                            percentage = (count / total_count * 100)
                            strategy_data.append({
                                '策略': strategy,
                                '使用次数': count,
                                '占比': f"{percentage:.1f}%"
                            })

                    strategy_df = pd.DataFrame(strategy_data)
                    st.dataframe(strategy_df, use_container_width=True, hide_index=True)

                    # 策略说明
                    with st.expander(t('fusion.strategy_explanation'), expanded=False):
                        st.markdown("""
                        **策略类型及其含义：**

                        - **ULTRA_AGGRESSIVE** (超激进): 任意1个模型同意即接受 - 用于高置信度+低冲突区域
                        - **AGGRESSIVE** (激进): ≥30%模型同意 - 用于高置信度+轻微冲突区域
                        - **RELAXED** (宽松): ≥40%模型同意 - 用于中等置信度+低冲突区域
                        - **STANDARD_HIGH_CONF** (标准-高置信): ≥50%模型同意 - 用于高置信度+中等冲突区域
                        - **STANDARD_MID_CONF** (标准-中置信): ≥50%模型同意 - 用于中等置信度+中等冲突区域
                        - **STANDARD_LOW_CONF** (标准-低置信): ≥50%模型同意 - 用于低置信度+低冲突区域
                        - **STRICT** (严格): ≥60%模型同意 - 用于中等置信度+高冲突区域
                        - **STRICT_LOW_CONF** (严格-低置信): ≥60%模型同意 - 用于低置信度+中等冲突区域
                        - **ULTRA_STRICT** (超严格): ≥70%模型同意 - 用于低置信度+高冲突区域
                        - **INTERSECTION** (交集): 100%模型同意 - 用于极端冲突区域

                        **策略选择基于二维决策矩阵：置信度 × 冲突度**
                        """)
                else:
                    st.warning(t('messages.strategy_distribution_not_found'))

                # 高冲突实例列表
                if dst_stats['high_conflict_count'] > 0:
                    st.markdown(t('fusion.high_conflict_instances_title'))
                    st.caption(t('fusion.high_conflict_note'))

                    import pandas as pd
                    conflict_df = pd.DataFrame(dst_stats['high_conflict_instances'])
                    st.dataframe(conflict_df, use_container_width=True)

                # 分水岭边界细化统计
                watershed_stats = dst_stats.get('watershed_refinement', {})
                if watershed_stats.get('refined', False):
                    st.markdown(t('messages.watershed_refinement_title'))
                    st.success(t('messages.watershed_refinement_success',
                                 pixels=watershed_stats['refined_pixels'],
                                 percentage=watershed_stats['refined_percentage']))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(t('metrics.seed_count'), watershed_stats.get('num_markers', 0))
                    with col2:
                        st.metric(t('metrics.high_conflict_pixels'), watershed_stats.get('high_conflict_pixels', 0))
                elif watershed_stats:
                    reason = watershed_stats.get('reason', 'unknown')
                    if reason == 'no_image':
                        st.info(t('messages.watershed_no_image'))
                    elif reason == 'no_markers':
                        st.warning(t('messages.watershed_no_seeds'))
                    elif reason == 'disabled_or_no_conflict':
                        st.info(t('messages.watershed_not_enabled'))
                    else:
                        st.warning(t('messages.watershed_refinement_failed', reason=reason))

                # 详细融合结果
                with st.expander(t('fusion.view_detailed_results'), expanded=False):
                    if len(dst_stats['fusion_results']) > 0:
                        import pandas as pd
                        results_df = pd.DataFrame(dst_stats['fusion_results'])
                        st.dataframe(results_df, use_container_width=True)

                        # 下载按钮
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label=t('fusion.download_dst_csv'),
                            data=csv,
                            file_name="dst_fusion_results.csv",
                            mime="text/csv"
                        )

        # 6. 统计信息
        st.subheader(t('common.statistics_summary'))
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(t('metrics.detected_cells'), np.max(fused_mask))
        with col2:
            st.metric(t('metrics.model_consistency'), f"{consistency_score:.2%}")
        with col3:
            st.metric(t('metrics.average_model_iou'), f"{avg_consistency:.2%}")

        # 7. 导出选项
        st.subheader(t('common.export_results'))

        # 创建导出选项卡
        export_tabs = st.tabs([t('fusion.export_mask_tab'), t('fusion.export_stats_tab'), t('fusion.export_viz_tab'), t('fusion.export_conflict_tab')])

        # Tab 1: 掩码导出
        with export_tabs[0]:
            st.markdown(f"#### {t('fusion.export_fusion_mask_title')}")

            col1, col2 = st.columns(2)

            with col1:
                # PNG格式导出（彩色可视化）
                from io import BytesIO
                from skimage.color import label2rgb

                fused_display = label2rgb(fused_mask, bg_label=0)
                fused_img = Image.fromarray((fused_display * 255).astype(np.uint8))
                buf_png = BytesIO()
                fused_img.save(buf_png, format='PNG')

                st.download_button(
                    label=t('fusion.download_png_color'),
                    data=buf_png.getvalue(),
                    file_name=f"fused_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
                st.caption(t('fusion.png_caption'))

            with col2:
                # TIFF格式导出（原始标签）
                from PIL import Image as PILImage

                # 将标签掩码转换为16位整数（支持更多标签）
                fused_mask_16bit = fused_mask.astype(np.uint16)
                mask_img = PILImage.fromarray(fused_mask_16bit)
                buf_tiff = BytesIO()
                mask_img.save(buf_tiff, format='TIFF')

                st.download_button(
                    label=t('fusion.download_tiff_label'),
                    data=buf_tiff.getvalue(),
                    file_name=f"fused_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff",
                    mime="image/tiff",
                    use_container_width=True
                )
                st.caption(t('fusion.tiff_caption'))

        # Tab 2: 统计报告导出
        with export_tabs[1]:
            st.markdown(f"#### {t('fusion.export_stats_title')}")

            col1, col2 = st.columns(2)

            with col1:
                # JSON格式导出
                import json

                # 准备导出数据
                export_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'basic_stats': {
                        'total_cells': int(np.max(fused_mask)),
                        'model_consistency': float(consistency_score),
                        'average_model_iou': float(avg_consistency)
                    },
                    'models_used': model_names
                }

                # 如果使用了DST融合，添加DST统计信息
                if dst_stats is not None:
                    export_data['dst_stats'] = {
                        'total_groups': dst_stats['total_groups'],
                        'fused_count': dst_stats['fused_count'],
                        'skipped_count': dst_stats['skipped_count'],
                        'high_conflict_count': dst_stats['high_conflict_count'],
                        'average_conflict': float(dst_stats['average_conflict']),
                        'average_uncertainty': float(dst_stats['average_uncertainty']),
                        'confidence_distribution': {k: float(v) for k, v in dst_stats['confidence_distribution'].items()},
                        'conflict_distribution': {k: float(v) for k, v in dst_stats['conflict_distribution'].items()},
                        'strategy_counts': dst_stats.get('strategy_counts', {})
                    }

                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

                st.download_button(
                    label=t('fusion.download_json'),
                    data=json_str,
                    file_name=f"fusion_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
                st.caption(t('fusion.json_caption'))

            with col2:
                # Excel格式导出
                import pandas as pd

                # 创建Excel writer
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Sheet 1: 基本统计
                    basic_df = pd.DataFrame({
                        '指标': ['检测细胞数', '模型一致性', '平均模型IoU', '使用模型数'],
                        '值': [
                            int(np.max(fused_mask)),
                            f"{consistency_score:.2%}",
                            f"{avg_consistency:.2%}",
                            len(model_names)
                        ]
                    })
                    basic_df.to_excel(writer, sheet_name='基本统计', index=False)

                    # Sheet 2: DST详细结果（如果有）
                    if dst_stats is not None and len(dst_stats['fusion_results']) > 0:
                        results_df = pd.DataFrame(dst_stats['fusion_results'])
                        results_df.to_excel(writer, sheet_name='DST融合结果', index=False)

                    # Sheet 3: 模型一致性矩阵
                    consistency_df = pd.DataFrame(
                        consistency_matrix,
                        columns=[f"模型{i+1}" for i in range(len(model_names))],
                        index=[f"模型{i+1}" for i in range(len(model_names))]
                    )
                    consistency_df.to_excel(writer, sheet_name='模型一致性矩阵')

                st.download_button(
                    label=t('fusion.download_excel'),
                    data=output.getvalue(),
                    file_name=f"fusion_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                st.caption(t('fusion.excel_caption'))

        # Tab 3: 可视化图像导出
        with export_tabs[2]:
            st.markdown(f"#### {t('fusion.export_viz_title')}")

            col1, col2 = st.columns(2)

            with col1:
                # 导出不确定性热图
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')

                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(disagreement_map, cmap='hot', interpolation='nearest')
                ax.set_title('Model Disagreement Heatmap', fontsize=14)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                buf_heatmap = BytesIO()
                plt.savefig(buf_heatmap, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig)

                st.download_button(
                    label=t('fusion.download_disagreement_heatmap'),
                    data=buf_heatmap.getvalue(),
                    file_name=f"disagreement_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
                st.caption(t('fusion.disagreement_caption'))

            with col2:
                # 导出模型一致性矩阵图
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(consistency_matrix, cmap='RdYlGn', vmin=0, vmax=1)
                ax.set_xticks(range(len(model_names)))
                ax.set_yticks(range(len(model_names)))
                ax.set_xticklabels([f"M{i+1}" for i in range(len(model_names))])
                ax.set_yticklabels([f"M{i+1}" for i in range(len(model_names))])
                ax.set_title('Model Consistency Matrix', fontsize=14)

                # 添加数值标注
                for i in range(len(model_names)):
                    for j in range(len(model_names)):
                        text = ax.text(j, i, f'{consistency_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=10)

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                buf_consistency = BytesIO()
                plt.savefig(buf_consistency, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig)

                st.download_button(
                    label=t('fusion.download_consistency_matrix'),
                    data=buf_consistency.getvalue(),
                    file_name=f"consistency_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
                st.caption(t('fusion.consistency_caption'))

        # Tab 4: 高冲突报告导出
        with export_tabs[3]:
            st.markdown(f"#### {t('fusion.export_conflict_title')}")

            if dst_stats is not None and dst_stats['high_conflict_count'] > 0:
                col1, col2 = st.columns(2)

                with col1:
                    # 导出高冲突实例列表（CSV）
                    import pandas as pd

                    conflict_df = pd.DataFrame(dst_stats['high_conflict_instances'])

                    csv_data = conflict_df.to_csv(index=False)

                    st.download_button(
                        label=t('fusion.download_conflict_csv'),
                        data=csv_data,
                        file_name=f"high_conflict_instances_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.caption(t('fusion.conflict_csv_caption', count=dst_stats['high_conflict_count']))

                with col2:
                    # 导出完整的冲突分析报告（Markdown）
                    report_lines = [
                        "# 模型融合冲突分析报告",
                        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"\n## 总体统计",
                        f"- 总实例组数: {dst_stats['total_groups']}",
                        f"- 成功融合: {dst_stats['fused_count']}",
                        f"- 跳过实例: {dst_stats['skipped_count']}",
                        f"- 高冲突实例: {dst_stats['high_conflict_count']}",
                        f"- 平均冲突度: {dst_stats['average_conflict']:.3f}",
                        f"- 平均不确定性: {dst_stats['average_uncertainty']:.3f}",
                        f"\n## 置信度分布",
                        f"- 最小值: {dst_stats['confidence_distribution']['min']:.3f}",
                        f"- 最大值: {dst_stats['confidence_distribution']['max']:.3f}",
                        f"- 平均值: {dst_stats['confidence_distribution']['mean']:.3f}",
                        f"- 标准差: {dst_stats['confidence_distribution']['std']:.3f}",
                        f"\n## 冲突度分布",
                        f"- 最小值: {dst_stats['conflict_distribution']['min']:.3f}",
                        f"- 最大值: {dst_stats['conflict_distribution']['max']:.3f}",
                        f"- 平均值: {dst_stats['conflict_distribution']['mean']:.3f}",
                        f"- 标准差: {dst_stats['conflict_distribution']['std']:.3f}",
                        f"\n## 高冲突实例详情",
                        "\n| 实例ID | 组索引 | 冲突度 | 不确定性 | 状态 |",
                        "|--------|--------|--------|----------|------|"
                    ]

                    for instance in dst_stats['high_conflict_instances']:
                        report_lines.append(
                            f"| {instance['instance_id']} | {instance['group_idx']} | "
                            f"{instance['conflict']:.3f} | {instance['uncertainty']:.3f} | "
                            f"{instance['status']} |"
                        )

                    report_text = "\n".join(report_lines)

                    st.download_button(
                        label=t('fusion.download_conflict_report'),
                        data=report_text,
                        file_name=f"conflict_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    st.caption(t('fusion.conflict_report_caption'))

            else:
                st.info(t('fusion.no_conflict_info'))
                st.caption(t('fusion.no_conflict_hint'))

    # 存储结果到session state，防止下载按钮导致页面重置
    st.session_state['fusion_results'] = {
        'fused_mask': fused_mask,
        'disagreement_map': disagreement_map,
        'consistency_matrix': consistency_matrix,
        'consistency_score': consistency_score,
        'avg_consistency': avg_consistency,
        'dst_stats': dst_stats,
        'model_names': model_names,
        'masks_list': masks_list,
        'image': image,
        'matched_groups': matched_groups,
        'postprocess_options': postprocess_options,
        'individual_cells': individual_cells,
        'cell_info': cell_info,
        'morphology_features': morphology_features
    }


# 页面标题和语言切换器
col_title, col_spacer, col_lang = st.columns([3, 1, 1])

with col_title:
    st.title(t('app.title'))

with col_lang:
    i18n = get_i18n()
    current_lang = i18n.get_current_language()

    selected_lang = st.radio(
        "🌐",
        options=['en_US', 'zh_CN'],
        format_func=lambda x: 'English' if x == 'en_US' else '中文',
        index=0 if current_lang == 'en_US' else 1,
        horizontal=True,
        key="language_radio",
        label_visibility="collapsed"
    )

    if selected_lang != current_lang:
        i18n.set_language(selected_lang)
        st.rerun()

# GPU状态显示
if GPU_AVAILABLE and GPU_COMPATIBLE:
    st.success(t('messages.gpu_available', gpu_name=GPU_NAME))
elif GPU_AVAILABLE and not GPU_COMPATIBLE:
    st.error(t('messages.gpu_incompatible', gpu_name=GPU_NAME, version=torch.__version__))
else:
    st.info(t('messages.gpu_unavailable'))

# 帮助文档
with st.expander(t('app.help_title'), expanded=False):
    st.markdown(f"### {t('help.overview.title')}")
    st.write(t('help.overview.content'))

    st.markdown("---")

    # 显示所有标签页的帮助信息
    for i in range(1, 11):
        tab_key = f'tab{i}'
        st.markdown(f"### {t(f'help.{tab_key}.title')}")
        st.write(t(f'help.{tab_key}.description'))

        st.markdown(f"**{t('help.features_label')}**")
        features = t(f'help.{tab_key}.features')
        if isinstance(features, list):
            for feature in features:
                st.markdown(f"- {feature}")

        st.markdown(f"**{t('help.usage_label')}**")
        st.write(t(f'help.{tab_key}.usage'))

        if i < 10:
            st.markdown("")

    st.markdown("---")

    # GPU加速说明
    st.markdown(f"### {t('help.gpu.title')}")
    st.write(t('help.gpu.description'))
    st.markdown(f"**{t('help.requirements_label')}**")
    requirements = t('help.gpu.requirements')
    if isinstance(requirements, list):
        for req in requirements:
            st.markdown(f"- {req}")
    st.info(t('help.gpu.performance'))

    st.markdown("---")

    # 提示与最佳实践
    st.markdown(f"### {t('help.tips.title')}")

    st.markdown(f"**{t('help.general_tips_label')}**")
    general_tips = t('help.tips.general')
    if isinstance(general_tips, list):
        for tip in general_tips:
            st.markdown(f"- {tip}")

    st.markdown(f"**{t('help.recommended_workflow_label')}**")
    workflow = t('help.tips.workflow')
    if isinstance(workflow, list):
        for step in workflow:
            st.markdown(f"- {step}")

    st.markdown(f"**{t('help.performance_tips_label')}**")
    performance = t('help.tips.performance')
    if isinstance(performance, list):
        for tip in performance:
            st.markdown(f"- {tip}")

# 创建标签页
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    t('tabs.image_segmentation'),
    t('tabs.comparison_mode'),
    t('tabs.model_fusion'),
    t('tabs.batch_processing'),
    t('tabs.cell_morphology'),
    t('tabs.anomaly_detection'),
    t('tabs.clustering_analysis'),
    t('tabs.supervised_learning'),
    t('tabs.active_learning'),
    t('tabs.virtual_screening')
])

# ==================== 标签页1: 图像分割 ====================
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.header(t('common.settings'))

        # 图像上传
        uploaded_file = st.file_uploader(
            t('common.upload_image'),
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="single_upload"
        )

        # 像素大小设置
        st.write(f"**{t('segmentation.pixel_size_setting')}**")
        pixel_size = st.number_input(
            t('segmentation.pixel_size'),
            min_value=0.01,
            max_value=10.0,
            value=0.65,
            step=0.01,
            help=t('segmentation.pixel_size_help')
        )
        st.session_state['pixel_size'] = pixel_size

        # 预处理选项
        with st.expander(t('segmentation.preprocessing'), expanded=False):
            denoise = st.checkbox(t('segmentation.denoise'), value=False, help=t('segmentation.denoise_help'))
            enhance = st.checkbox(t('segmentation.enhance'), value=False, help=t('segmentation.enhance_help'))
            normalize = st.checkbox(t('segmentation.normalize'), value=False, help=t('segmentation.normalize_help'))

        # 后处理选项
        with st.expander(t('segmentation.postprocessing'), expanded=False):
            closing = st.checkbox(t('segmentation.region_closing'), value=True, help=t('segmentation.region_closing_help'))
            if closing:
                closing_kernel_size = st.slider(t('segmentation.closing_kernel_size'), 3, 15, 3, 2, help=t('segmentation.closing_kernel_help'))
            else:
                closing_kernel_size = 5

            extract_cells = st.checkbox(t('segmentation.extract_cells'), value=False, help=t('segmentation.extract_cells_help'))
            if extract_cells:
                min_cell_area = st.slider(t('segmentation.min_cell_area'), 50, 500, 50, 10, help=t('segmentation.min_cell_area_help'))

            extract_morphology = st.checkbox(t('segmentation.extract_features'), value=False, help=t('segmentation.extract_features_help'))

            # 高级特征提取选项
            use_advanced_features = st.checkbox(t('segmentation.advanced_features'), value=False, help=t('segmentation.advanced_features_help'))
            if use_advanced_features:
                st.caption(f"**{t('segmentation.advanced_feature_categories')}**")
                include_hu_moments = st.checkbox(t('segmentation.hu_moments'), value=True, help=t('segmentation.hu_moments_help'))
                include_intensity = st.checkbox(t('segmentation.intensity_stats'), value=True, help=t('segmentation.intensity_stats_help'))
                include_texture = st.checkbox(t('segmentation.texture_glcm'), value=True, help=t('segmentation.texture_glcm_help'))
                include_boundary = st.checkbox(t('segmentation.boundary_complexity'), value=True, help=t('segmentation.boundary_complexity_help'))
                include_advanced_shape = st.checkbox(t('segmentation.advanced_shape'), value=True, help=t('segmentation.advanced_shape_help'))

        # 分割方法选择
        st.subheader(t('segmentation.method'))
        method = st.selectbox(
            t('segmentation.select_method'),
            [t('methods.otsu'), t('methods.adaptive'), t('methods.watershed'),
             t('methods.canny'), t('methods.cellpose'), t('methods.cellvit'), t('methods.cellsam')],
            index=4  # 默认选择Cellpose深度学习
        )

        # 方法参数（直接显示，不使用折叠面板）
        if method == t('methods.adaptive'):
            st.write(f"**{t('segmentation.method_params')}**")
            block_size = st.slider(t('segmentation.block_size'), 3, 51, 11, 2)
            C = st.slider(t('segmentation.constant_c'), 0, 20, 2)
            params = {"block_size": block_size, "C": C}
        elif method == t('methods.canny'):
            st.write(f"**{t('segmentation.method_params')}**")
            low_threshold = st.slider(t('segmentation.low_threshold'), 0, 200, 50, 10)
            high_threshold = st.slider(t('segmentation.high_threshold'), 0, 300, 150, 10)
            params = {"low_threshold": low_threshold, "high_threshold": high_threshold}
        elif method == t('methods.cellpose'):
            st.write(f"**{t('segmentation.method_params')}**")
            model_type = st.selectbox(t('segmentation.model_type'), ["cyto2", "cyto", "nuclei"],
                                     help=t('segmentation.model_type_help'))
            diameter = st.slider(t('segmentation.cell_diameter'), 0, 100, 30, 5,
                                help=t('segmentation.cell_diameter_help'))
            if diameter == 0:
                diameter = None

            # GPU选项
            if GPU_AVAILABLE and GPU_COMPATIBLE:
                use_gpu = st.checkbox(t('segmentation.use_gpu', gpu_name=GPU_NAME), value=True,
                                     help=t('segmentation.use_gpu_help'))
            elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                use_gpu = False
                st.error(t('segmentation.gpu_incompatible', warning=GPU_WARNING))
                st.info(t('segmentation.gpu_cpu_mode'))
            else:
                use_gpu = False
                st.info(t('messages.gpu_unavailable'))

            # 高级参数
            with st.expander(t('segmentation.advanced_params'), expanded=False):
                batch_size = st.slider(t('segmentation.batch_size'), 1, 64, 8, 1,
                                      help=t('segmentation.batch_size_help'))
                use_normalize = st.checkbox(t('segmentation.enable_normalize'), value=True,
                                           help=t('segmentation.enable_normalize_help'))
                if use_normalize:
                    tile_norm_blocksize = st.slider(t('segmentation.normalize_block_size'), 0, 256, 64, 16,
                                                   help=t('segmentation.normalize_block_size_help'))
                    normalize = {"tile_norm_blocksize": tile_norm_blocksize}
                else:
                    normalize = None

            params = {"model_type": model_type, "diameter": diameter, "use_gpu": use_gpu,
                     "batch_size": batch_size, "normalize": normalize}
        elif method == t('methods.cellvit'):
            st.write(t('segmentation.method_params_title'))

            # 环境检查
            if not CELLVIT_ENV_OK:
                st.error(t('messages.cellvit_env_not_found'))
                st.warning(t('messages.cellvit_env_required'))
                st.code("conda create --prefix ./env_cellvit python=3.12 -y\nsource activate ./env_cellvit\npip install cellvit torch torchvision", language="bash")
                st.info(t('messages.cellvit_alternative'))
            else:
                st.success(t('messages.cellvit_env_ready').format(env=CELLVIT_ENV_STATUS))

            model_type = st.selectbox(t('segmentation.model_type'), ["CellViT-256"],
                                     help=t('segmentation.model_type_help'))
            target_size = st.slider(t('segmentation.target_size'), 256, 1024, 512, 64,
                                   help=t('segmentation.target_size_help'))

            # GPU选项
            if GPU_AVAILABLE and GPU_COMPATIBLE:
                use_gpu = st.checkbox(t('segmentation.use_gpu').format(gpu_name=GPU_NAME), value=True,
                                     help=t('messages.cellvit_recommended_gpu'))
            elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                use_gpu = False
                st.error(t('segmentation.gpu_incompatible').format(warning=GPU_WARNING))
                st.info(t('messages.gpu_cpu_mode'))
            else:
                use_gpu = False
                st.info(t('messages.gpu_unavailable'))

            params = {"model_type": model_type, "target_size": target_size, "use_gpu": use_gpu}
        elif method == t('methods.cellsam'):
            st.write(t('segmentation.method_params_title'))

            # 提示信息
            st.info(t('messages.cellsam_info'))

            model_type = st.selectbox(t('segmentation.model_type'), ["vit_b", "vit_l", "vit_h"],
                                     help=t('segmentation.sam_model_type_help'))
            points_per_side = st.slider(t('segmentation.points_per_side'), 16, 64, 32, 8,
                                       help=t('segmentation.points_per_side_help'))

            # GPU选项
            if GPU_AVAILABLE and GPU_COMPATIBLE:
                use_gpu = st.checkbox(t('segmentation.use_gpu').format(gpu_name=GPU_NAME), value=True,
                                     help=t('messages.cellsam_recommended_gpu'))
            elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                use_gpu = False
                st.error(t('segmentation.gpu_incompatible').format(warning=GPU_WARNING))
                st.info(t('messages.gpu_cpu_mode'))
            else:
                use_gpu = False
                st.info(t('messages.gpu_unavailable'))

            params = {"model_type": model_type, "points_per_side": points_per_side, "use_gpu": use_gpu}
        else:
            params = {}

        segment_btn = st.button(t('common.start_processing'), type="primary", use_container_width=True)

    with col_right:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            st.subheader(t('common.original_image'))
            st.image(image, width=400)
            st.caption(t('common.image_size', shape=image_np.shape))

            if segment_btn:
                st.subheader(t('common.segmentation_result'))

                # 根据方法显示不同的进度提示
                spinner_text = t('common.processing')
                if method == t('methods.cellpose'):
                    spinner_text = t('messages.processing_cellpose')

                with st.spinner(spinner_text):
                    try:
                        preprocess_options = {
                            'denoise': denoise,
                            'enhance': enhance,
                            'normalize': normalize
                        }

                        postprocess_options = {
                            'closing': closing,
                            'closing_kernel_size': closing_kernel_size,
                            'extract_cells': extract_cells,
                            'min_cell_area': min_cell_area if extract_cells else 100,
                            'extract_morphology': extract_morphology
                        }

                        # 为Cellpose创建进度条
                        if method == t('methods.cellpose'):
                            cellpose_progress = st.progress(0)
                            st.caption(t('common.cellpose_progress_caption'))
                            params['progress_bar'] = cellpose_progress

                        result = segment_single_image(image_np, method, params, preprocess_options, postprocess_options)

                        # 清除Cellpose进度条
                        if method == t('methods.cellpose'):
                            cellpose_progress.empty()

                        # 显示结果
                        tab_mask, tab_overlay = st.tabs([t('common.mask_display'), t('common.overlay_display')])

                        with tab_mask:
                            st.image(result['mask'], use_container_width=True)

                        with tab_overlay:
                            st.image(result['overlay'], use_container_width=True)

                        # 统计信息
                        st.success(t('common.segmentation_completed'))

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric(t('metrics.foreground_pixels'), f"{result['foreground_pixels']:,}")
                        with col_b:
                            st.metric(t('metrics.foreground_ratio'), f"{result['foreground_ratio']:.2f}%")
                        with col_c:
                            st.metric(t('metrics.processing_time'), f"{result['processing_time']*1000:.2f} ms")

                        if result['num_regions'] is not None:
                            st.info(t('messages.cells_detected', count=result['num_regions']))

                        # 细胞形态学特征提取
                        if (extract_morphology or use_advanced_features) and result['num_regions'] is not None and result['num_regions'] > 0:
                            st.subheader(t('common.morphology_analysis'))

                            with st.spinner(t('common.extracting_features')):
                                # 提取特征（使用与单个细胞提取相同的min_area过滤）
                                pixel_size = st.session_state.get('pixel_size', 1.0)
                                min_area = postprocess_options.get('min_cell_area', 100)

                                # 根据用户选择调用不同的特征提取函数
                                if use_advanced_features:
                                    # 使用高级特征提取（需要原始图像）
                                    # 将图像转换为灰度图（如果是彩色的）
                                    if len(image_np.shape) == 3:
                                        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                                    else:
                                        gray_image = image_np

                                    features_df = extract_advanced_cell_features(
                                        result['labeled_mask'],
                                        image=gray_image,
                                        pixel_size=pixel_size,
                                        min_area=min_area,
                                        include_hu_moments=include_hu_moments,
                                        include_intensity=include_intensity,
                                        include_texture=include_texture,
                                        include_boundary=include_boundary,
                                        include_advanced_shape=include_advanced_shape
                                    )
                                    st.success(t('messages.advanced_features_completed'))
                                else:
                                    # 使用基础特征提取
                                    features_df = extract_cell_features(result['labeled_mask'], pixel_size=pixel_size, min_area=min_area)

                                if not features_df.empty:
                                    # 显示特征统计
                                    st.write(t('common.feature_statistics_title'))
                                    stats = get_feature_statistics(features_df)

                                    # 显示关键特征的统计信息
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric(t('metrics.average_area'), f"{stats['area_um2']['mean']:.1f} μm²")
                                    with col2:
                                        st.metric(t('metrics.average_circularity'), f"{stats['circularity']['mean']:.3f}")
                                    with col3:
                                        st.metric(t('metrics.average_major_axis'), f"{stats['major_axis_length']['mean']:.1f} μm")
                                    with col4:
                                        st.metric(t('metrics.average_minor_axis'), f"{stats['minor_axis_length']['mean']:.1f} μm")

                                    # 显示详细特征表格
                                    with st.expander(t('common.view_detailed_features'), expanded=False):
                                        # ID说明
                                        st.info("""
                                        **📌 关于细胞ID：**
                                        - **sequential_id**：连续编号（1, 2, 3...），用于数据分析和统计
                                        - **cell_id**：原始分割mask中的标签ID（可能不连续），用于追溯原始分割结果

                                        **为什么cell_id不连续？** 因为面积小于阈值的细胞被过滤掉了，但它们的标签ID仍保留在原始mask中。

                                        **如何对照？**
                                        - 查看单个细胞图像时，使用 **cell_id** 在原始mask中定位
                                        - 进行数据分析时，使用 **sequential_id** 作为连续索引
                                        """)

                                        # 选择要显示的列
                                        if use_advanced_features:
                                            # 高级特征模式：显示所有列
                                            st.caption(t('morphology.table_scroll_hint'))
                                            st.dataframe(features_df.round(3), use_container_width=True, height=400)
                                        else:
                                            # 基础特征模式：只显示主要列
                                            display_cols = ['sequential_id', 'cell_id', 'area_um2', 'perimeter_um', 'circularity',
                                                          'major_axis_length', 'minor_axis_length', 'eccentricity',
                                                          'solidity', 'aspect_ratio']
                                            st.dataframe(features_df[display_cols].round(3), use_container_width=True, height=400)

                                    # 保存特征数据到session_state供导出使用
                                    st.session_state['cell_features'] = features_df

                        # 导出按钮
                        col_export1, col_export2, col_export3 = st.columns(3)

                        with col_export1:
                            # 导出掩码
                            mask_bytes = cv2.imencode('.png', result['mask'])[1].tobytes()
                            st.download_button(
                                "💾 下载掩码",
                                data=mask_bytes,
                                file_name=f"mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )

                        with col_export2:
                            # 导出叠加图
                            overlay_bytes = cv2.imencode('.png', result['overlay'])[1].tobytes()
                            st.download_button(
                                "💾 下载叠加图",
                                data=overlay_bytes,
                                file_name=f"overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )

                        with col_export3:
                            # 导出特征数据
                            if 'cell_features' in st.session_state and not st.session_state['cell_features'].empty:
                                csv_data = st.session_state['cell_features'].to_csv(index=False)
                                st.download_button(
                                    t('common.download_features'),
                                    data=csv_data,
                                    file_name=f"cell_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )

                        # 单个细胞提取结果
                        if result['individual_cells'] is not None and len(result['individual_cells']) > 0:
                            st.subheader(t('common.individual_cells'))
                            st.info(t('messages.successfully_extracted_cells', count=len(result['individual_cells'])))

                            # 显示前几个细胞样本
                            st.write(t('common.sample_preview'))
                            cols_preview = st.columns(6)
                            for idx, cell_data in enumerate(result['individual_cells'][:6]):
                                with cols_preview[idx]:
                                    st.image(cell_data['image'], caption=t('common.cell_number', number=idx+1), use_container_width=True)

                            # 导出单个细胞样本
                            cells_zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(cells_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for idx, cell_data in enumerate(result['individual_cells']):
                                    # 保存细胞图像
                                    cell_img_bytes = cv2.imencode('.png', cell_data['image'])[1].tobytes()
                                    zip_file.writestr(f"cell_{idx+1:03d}_image.png", cell_img_bytes)

                                    # 保存细胞掩码
                                    cell_mask_bytes = cv2.imencode('.png', cell_data['mask'])[1].tobytes()
                                    zip_file.writestr(f"cell_{idx+1:03d}_mask.png", cell_mask_bytes)

                                # 保存细胞信息CSV
                                if result['cell_info']:
                                    cell_df = pd.DataFrame(result['cell_info'])
                                    csv_buffer = io.StringIO()
                                    cell_df.to_csv(csv_buffer, index=False)
                                    zip_file.writestr("cells_info.csv", csv_buffer.getvalue())

                            st.download_button(
                                t('common.download_all_cells'),
                                data=cells_zip_buffer.getvalue(),
                                file_name=f"cells_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"❌ {t('common.error')}: {str(e)}")
        else:
            st.info(t('common.please_upload_left'))

# ==================== 标签页2: 对比模式 ====================
with tab2:
    st.header(t('tabs.comparison_mode'))
    st.caption(t('comparison.description'))

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader(t('common.settings'))

        # 图像上传
        uploaded_file = st.file_uploader(
            t('common.upload_image'),
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="comparison_upload"
        )

        # 像素大小设置
        st.write(f"**{t('segmentation.pixel_size_setting')}**")
        pixel_size = st.number_input(
            t('segmentation.pixel_size'),
            min_value=0.01,
            max_value=10.0,
            value=0.65,
            step=0.01,
            help=t('segmentation.pixel_size_help'),
            key="comparison_pixel_size"
        )
        st.session_state['pixel_size'] = pixel_size

        # 预处理选项
        with st.expander(t('segmentation.preprocessing'), expanded=False):
            denoise = st.checkbox(t('segmentation.denoise'), value=False, help=t('segmentation.denoise_help'), key="comparison_denoise")
            enhance = st.checkbox(t('segmentation.enhance'), value=False, help=t('segmentation.enhance_help'), key="comparison_enhance")
            normalize = st.checkbox(t('segmentation.normalize'), value=False, help=t('segmentation.normalize_help'), key="comparison_normalize")

        # 后处理选项
        with st.expander(t('segmentation.postprocessing'), expanded=False):
            closing = st.checkbox(t('segmentation.region_closing'), value=True, help=t('segmentation.region_closing_help'), key="comparison_closing")
            if closing:
                closing_kernel_size = st.slider(t('segmentation.closing_kernel_size'), 3, 15, 3, 2, help=t('segmentation.closing_kernel_help'), key="comparison_closing_size")
            else:
                closing_kernel_size = 5

            extract_cells = st.checkbox(t('segmentation.extract_cells'), value=False, help=t('segmentation.extract_cells_help'), key="comparison_extract_cells")
            if extract_cells:
                min_cell_area = st.slider(t('segmentation.min_cell_area'), 50, 500, 50, 10, help=t('segmentation.min_cell_area_help'), key="comparison_min_area")
            else:
                min_cell_area = 100

            extract_morphology = st.checkbox(t('segmentation.extract_features'), value=False, help=t('segmentation.extract_features_help'), key="comparison_extract_morph")

        # 对比方法选择
        st.subheader(t('common.comparison_method_selection'))
        st.write(t('common.select_methods_to_compare'))
        comp_methods = []
        if st.checkbox(t('methods.otsu'), value=True, key="comp_otsu_tab2"):
            comp_methods.append(t('methods.otsu'))
        if st.checkbox(t('methods.adaptive'), value=True, key="comp_adaptive_tab2"):
            comp_methods.append(t('methods.adaptive'))
        if st.checkbox(t('methods.watershed'), value=False, key="comp_watershed_tab2"):
            comp_methods.append(t('methods.watershed'))
        if st.checkbox(t('methods.canny'), value=False, key="comp_canny_tab2"):
            comp_methods.append(t('methods.canny'))
        if st.checkbox(t('methods.cellpose'), value=False, key="comp_cellpose_tab2"):
            comp_methods.append(t('methods.cellpose'))
        if st.checkbox(t('methods.cellvit'), value=False, key="comp_cellvit_tab2"):
            comp_methods.append(t('methods.cellvit'))
        if st.checkbox(t('methods.cellsam'), value=False, key="comp_cellsam_tab2"):
            comp_methods.append(t('methods.cellsam'))

        segment_btn = st.button(t('common.start_comparison'), type="primary", use_container_width=True, key="comparison_segment_btn")

    with col_right:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            st.subheader(t('common.original_image'))
            st.image(image, width=400)
            st.caption(t('common.image_size', shape=image_np.shape))

            if segment_btn:
                st.subheader(t('common.comparison_results'))

                if len(comp_methods) < 2:
                    st.warning(t('messages.select_at_least_two_methods'))
                else:
                    with st.spinner(t('common.processing_comparison')):
                        try:
                            preprocess_options = {
                                'denoise': denoise,
                                'enhance': enhance,
                                'normalize': normalize
                            }

                            postprocess_options = {
                                'closing': closing,
                                'closing_kernel_size': closing_kernel_size,
                                'extract_cells': extract_cells,
                                'min_cell_area': min_cell_area,
                                'extract_morphology': extract_morphology
                            }

                            # 为每种方法设置默认参数
                            params_dict = {
                                t('methods.otsu'): {},
                                t('methods.adaptive'): {"block_size": 11, "C": 2},
                                t('methods.watershed'): {},
                                t('methods.canny'): {"low_threshold": 50, "high_threshold": 150},
                                t('methods.cellpose'): {
                                    "model_type": "cyto2",
                                    "diameter": None,
                                    "use_gpu": (GPU_AVAILABLE and GPU_COMPATIBLE),
                                    "batch_size": 8,
                                    "normalize": {"tile_norm_blocksize": 0}
                                },
                                t('methods.cellvit'): {
                                    "model_type": "CellViT-256",
                                    "target_size": 768,
                                    "use_gpu": (GPU_AVAILABLE and GPU_COMPATIBLE)
                                },
                                t('methods.cellsam'): {
                                    "model_type": "vit_b",
                                    "points_per_side": 32,
                                    "use_gpu": (GPU_AVAILABLE and GPU_COMPATIBLE)
                                }
                            }

                            # 执行对比
                            comparison_results = create_comparison_view(
                                image_np,
                                comp_methods,
                                params_dict,
                                preprocess_options,
                                postprocess_options
                            )

                            # 显示对比结果
                            num_methods = len(comp_methods)
                            cols = st.columns(num_methods)

                            for idx, method_name in enumerate(comp_methods):
                                with cols[idx]:
                                    st.write(f"**{method_name}**")
                                    result = comparison_results[method_name]

                                    # 显示掩码
                                    st.image(result['mask'], use_container_width=True)

                                    # 显示统计
                                    st.metric(t('metrics.foreground_ratio'), f"{result['foreground_ratio']:.2f}%")
                                    st.metric(t('metrics.processing_time'), f"{result['processing_time']*1000:.2f} ms")

                            # 性能对比表
                            st.subheader(t('common.performance_comparison'))

                            comp_df_data = []
                            for method_name in comp_methods:
                                result = comparison_results[method_name]
                                comp_df_data.append({
                                    t('common.method_column'): method_name,
                                    t('metrics.foreground_pixels'): result['foreground_pixels'],
                                    f"{t('metrics.foreground_ratio')}(%)": f"{result['foreground_ratio']:.2f}",
                                    f"{t('metrics.processing_time')}(ms)": f"{result['processing_time']*1000:.2f}"
                                })

                            comp_df = pd.DataFrame(comp_df_data)
                            st.dataframe(comp_df, use_container_width=True)

                            # 推荐最快的方法
                            fastest_method = min(comp_methods, key=lambda m: comparison_results[m]['processing_time'])
                            st.success(t('common.fastest_method', method=fastest_method, time=comparison_results[fastest_method]['processing_time']*1000))

                            # 对比结果导出
                            st.subheader(t('common.export_results'))

                            col_comp_export1, col_comp_export2, col_comp_export3 = st.columns(3)

                            # 准备ZIP文件（所有方法的掩码）
                            comp_zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(comp_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for method_name in comp_methods:
                                    result = comparison_results[method_name]
                                    mask_bytes = cv2.imencode('.png', result['mask'])[1].tobytes()
                                    filename = f"mask_{method_name.replace('阈值', '').replace('算法', '').replace('检测', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                                    zip_file.writestr(filename, mask_bytes)

                            # 准备ZIP文件（所有方法的叠加图）
                            comp_overlay_zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(comp_overlay_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for method_name in comp_methods:
                                    result = comparison_results[method_name]
                                    overlay_bytes = cv2.imencode('.png', result['overlay'])[1].tobytes()
                                    filename = f"overlay_{method_name.replace('阈值', '').replace('算法', '').replace('检测', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                                    zip_file.writestr(filename, overlay_bytes)

                            # 准备CSV文件（对比统计）
                            comp_csv_buffer = io.BytesIO()
                            comp_df.to_csv(comp_csv_buffer, index=False, encoding='gbk')

                            with col_comp_export1:
                                st.download_button(
                                    t('common.export_all_masks'),
                                    data=comp_zip_buffer.getvalue(),
                                    file_name=f"comparison_masks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )

                            with col_comp_export2:
                                st.download_button(
                                    t('common.export_all_overlays'),
                                    data=comp_overlay_zip_buffer.getvalue(),
                                    file_name=f"comparison_overlays_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )

                            with col_comp_export3:
                                st.download_button(
                                    t('common.export_comparison_report'),
                                    data=comp_csv_buffer.getvalue(),
                                    file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )

                        except Exception as e:
                            st.error(f"❌ {t('common.error')}: {str(e)}")
        else:
            st.info(t('common.please_upload_left'))

# ==================== 标签页3: 模型融合 ====================
with tab3:
    st.header(t('tabs.model_fusion'))
    st.markdown(t('fusion.description'))

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader(t('common.image_upload'))
        fusion_uploaded = st.file_uploader(
            t('common.upload_image'),
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            key="fusion_upload"
        )

        if fusion_uploaded:
            fusion_image = Image.open(fusion_uploaded)
            fusion_image_np = np.array(fusion_image)

            st.subheader(t('common.model_selection'))
            st.markdown(t('fusion.select_models'))

            # 深度学习模型
            st.markdown(t('fusion.deep_learning_models'))
            use_cellpose = st.checkbox(t('methods.cellpose'), value=True, key="fusion_cellpose")
            use_cellvit = st.checkbox(t('methods.cellvit'), value=False, key="fusion_cellvit")
            use_cellsam = st.checkbox(t('methods.cellsam'), value=True, key="fusion_cellsam")

            # 传统方法
            st.markdown(t('fusion.traditional_methods'))
            use_watershed = st.checkbox(t('methods.watershed'), value=False, key="fusion_watershed")
            use_otsu = st.checkbox(t('methods.otsu'), value=False, key="fusion_otsu")
            use_adaptive = st.checkbox(t('methods.adaptive'), value=False, key="fusion_adaptive")
            use_canny = st.checkbox(t('methods.canny'), value=False, key="fusion_canny")

            selected_models = []
            if use_cellpose: selected_models.append("cellpose")
            if use_cellvit: selected_models.append("cellvit")
            if use_cellsam: selected_models.append("cellsam")
            if use_watershed: selected_models.append("watershed")
            if use_otsu: selected_models.append("otsu")
            if use_adaptive: selected_models.append("adaptive")
            if use_canny: selected_models.append("canny")

            if len(selected_models) < 2:
                st.warning(t('messages.select_at_least_two_models'))
            else:
                # 模型参数配置
                with st.expander(t('fusion.model_params_config'), expanded=False):
                    model_params = {}

                    if use_cellpose:
                        st.markdown(t('fusion.cellpose_params'))
                        model_params['cellpose_diameter'] = st.slider(t('segmentation.cell_diameter'), 10, 100, 30, key="fusion_cp_dia")
                        model_params['cellpose_gpu'] = st.checkbox(t('segmentation.use_gpu', gpu_name=GPU_NAME if GPU_AVAILABLE else "N/A"), value=True, key="fusion_cp_gpu")

                    if use_cellvit:
                        st.markdown(t('fusion.cellvit_params'))
                        model_params['cellvit_size'] = st.selectbox(t('segmentation.model_type'), [256, 512], index=0, key="fusion_cv_size")
                        model_params['cellvit_gpu'] = st.checkbox(t('segmentation.use_gpu', gpu_name=GPU_NAME if GPU_AVAILABLE else "N/A"), value=True, key="fusion_cv_gpu")

                    if use_cellsam:
                        st.markdown(t('fusion.cellsam_params'))
                        model_params['cellsam_points'] = st.slider(t('segmentation.points_per_side'), 16, 64, 32, key="fusion_sam_points")
                        model_params['cellsam_gpu'] = st.checkbox(t('segmentation.use_gpu', gpu_name=GPU_NAME if GPU_AVAILABLE else "N/A"), value=True, key="fusion_sam_gpu")

                    if use_watershed:
                        st.markdown(t('fusion.watershed_params'))
                        model_params['watershed_min_distance'] = st.slider(t('fusion.min_distance'), 5, 30, 10, key="fusion_ws_dist")
                        model_params['watershed_threshold'] = st.slider(t('fusion.threshold'), 0.3, 0.9, 0.5, 0.05, key="fusion_ws_thresh")

                    if use_otsu:
                        st.markdown(t('fusion.otsu_params'))
                        st.info(t('messages.otsu_auto_threshold'))

                    if use_adaptive:
                        st.markdown(t('fusion.adaptive_params'))
                        model_params['adaptive_block_size'] = st.slider(t('segmentation.block_size'), 11, 51, 21, 2, key="fusion_adp_block")
                        model_params['adaptive_c'] = st.slider(t('segmentation.constant_c'), 0, 20, 5, key="fusion_adp_c")

                    if use_canny:
                        st.markdown(t('fusion.canny_params'))
                        model_params['canny_threshold1'] = st.slider(t('segmentation.low_threshold'), 20, 150, 50, key="fusion_canny_t1")
                        model_params['canny_threshold2'] = st.slider(t('segmentation.high_threshold'), 50, 300, 150, key="fusion_canny_t2")

                # 融合策略选择
                st.subheader(t('common.fusion_strategy'))

                # 策略类型选择
                strategy_type = st.radio(
                    t('fusion.strategy_type'),
                    [t('fusion.simple_strategy'), t('fusion.advanced_dst')],
                    key="strategy_type"
                )

                if strategy_type == t('fusion.simple_strategy'):
                    fusion_strategy = st.radio(
                        t('fusion.select_fusion_strategy'),
                        ["majority", "weighted", "union", "intersection"],
                        format_func=lambda x: {
                            "majority": t('fusion.simple_voting'),
                            "weighted": t('fusion.weighted_voting'),
                            "union": t('fusion.aggressive_fusion'),
                            "intersection": t('fusion.conservative_fusion')
                        }[x],
                        key="fusion_strategy"
                    )
                else:
                    fusion_strategy = "dempster_shafer"
                    st.info(t('messages.dst_fusion_info'))

                # 高级选项
                with st.expander(t('fusion.advanced_options'), expanded=False):
                    iou_threshold = st.slider(t('fusion.iou_threshold'), 0.2, 0.9, 0.2, 0.05, key="fusion_iou")

                    # 最小投票数设置（只在3个或更多模型时显示slider）
                    if len(selected_models) == 2:
                        st.info(t('messages.min_vote_count_fixed'))
                        min_vote_count = 2
                    else:
                        min_vote_count = st.slider(t('fusion.min_vote_count'), 2, len(selected_models), 2, key="fusion_min_vote")

                    if fusion_strategy == "weighted":
                        st.markdown(t('fusion.model_weights'))
                        weights = {}
                        if use_cellpose:
                            weights['cellpose'] = st.slider(t('fusion.cellpose_weight'), 0.1, 2.0, 1.0, 0.1, key="w_cp")
                        if use_cellvit:
                            weights['cellvit'] = st.slider(t('fusion.cellvit_weight'), 0.1, 2.0, 1.0, 0.1, key="w_cv")
                        if use_cellsam:
                            weights['cellsam'] = st.slider(t('fusion.cellsam_weight'), 0.1, 2.0, 1.0, 0.1, key="w_sam")
                    else:
                        weights = None

                    # DST特定参数
                    if fusion_strategy == "dempster_shafer":
                        st.markdown(t('fusion.dst_reliability_params'))
                        st.caption(t('fusion.reliability_description'))

                        model_reliabilities = {}
                        if use_cellpose:
                            model_reliabilities['cellpose'] = st.slider(t('fusion.cellpose_reliability'), 0.5, 1.0, 0.9, 0.05, key="dst_r_cp")
                        if use_cellvit:
                            model_reliabilities['cellvit'] = st.slider(t('fusion.cellvit_reliability'), 0.5, 1.0, 0.85, 0.05, key="dst_r_cv")
                        if use_cellsam:
                            model_reliabilities['cellsam'] = st.slider(t('fusion.cellsam_reliability'), 0.5, 1.0, 0.8, 0.05, key="dst_r_sam")

                        # 传统方法的可靠性
                        if 'watershed' in selected_models:
                            model_reliabilities['watershed'] = st.slider(t('fusion.watershed_reliability'), 0.5, 1.0, 0.7, 0.05, key="dst_r_ws")
                        if 'otsu' in selected_models:
                            model_reliabilities['otsu'] = st.slider(t('fusion.otsu_reliability'), 0.5, 1.0, 0.65, 0.05, key="dst_r_otsu")
                        if 'adaptive' in selected_models:
                            model_reliabilities['adaptive'] = st.slider(t('fusion.adaptive_reliability'), 0.5, 1.0, 0.65, 0.05, key="dst_r_adp")
                        if 'canny' in selected_models:
                            model_reliabilities['canny'] = st.slider(t('fusion.canny_reliability'), 0.5, 1.0, 0.6, 0.05, key="dst_r_canny")

                        conflict_threshold = st.slider(t('fusion.conflict_threshold'), 0.3, 0.9, 0.4, 0.05, key="dst_conflict_th",
                                                      help=t('fusion.conflict_threshold_help'))
                    else:
                        model_reliabilities = None
                        conflict_threshold = 0.4

                # 后处理选项
                with st.expander(t('segmentation.postprocessing_options'), expanded=False):
                    st.markdown(t('segmentation.cell_extraction_analysis'))

                    fusion_extract_cells = st.checkbox(
                        t('segmentation.extract_individual_cells'),
                        value=False,
                        key="fusion_extract_cells",
                        help=t('segmentation.extract_cells_help')
                    )

                    fusion_min_cell_area = st.slider(
                        t('segmentation.min_cell_area'),
                        50, 500, 100,
                        key="fusion_min_cell_area",
                        help=t('segmentation.min_cell_area_help')
                    )

                    fusion_extract_morphology = st.checkbox(
                        t('segmentation.extract_morphology_features'),
                        value=False,
                        key="fusion_extract_morphology",
                        help=t('segmentation.extract_features_help')
                    )

                    fusion_use_advanced_features = st.checkbox(
                        t('segmentation.advanced_features'),
                        value=False,
                        key="fusion_use_advanced_features",
                        help=t('segmentation.advanced_features_help')
                    )

                # 开始融合按钮
                if st.button(t('common.start_fusion'), type="primary", key="start_fusion"):
                    # 构建后处理选项
                    fusion_postprocess_options = {
                        'closing': True,
                        'closing_kernel_size': 5,
                        'extract_cells': fusion_extract_cells,
                        'min_cell_area': fusion_min_cell_area,
                        'extract_morphology': fusion_extract_morphology,
                        'use_advanced_features': fusion_use_advanced_features
                    }

                    run_fusion_pipeline(
                        fusion_image_np, selected_models, fusion_strategy,
                        iou_threshold, min_vote_count, weights, model_params, col_right,
                        model_reliabilities=model_reliabilities, conflict_threshold=conflict_threshold,
                        postprocess_options=fusion_postprocess_options
                    )

    with col_right:
        if fusion_uploaded:
            st.subheader(t('common.original_image'))
            st.image(fusion_image, width=400)
            st.caption(t('common.image_size', shape=fusion_image_np.shape))

            # 融合结果将由run_fusion_pipeline函数在此列中显示
            # 不需要重复显示逻辑，避免"融合结果"标题重复出现
        else:
            st.info(t('messages.upload_image_left'))

# ==================== 标签页4: 批量处理 ====================
with tab4:
    st.header(t('tabs.batch_processing'))

    col_batch_left, col_batch_right = st.columns([1, 2])

    with col_batch_left:
        st.subheader(t('common.batch_settings'))

        # 批量上传
        uploaded_files = st.file_uploader(
            t('common.upload_multiple_images'),
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key="batch_upload"
        )

        # 预处理选项
        with st.expander(t('segmentation.preprocessing')):
            batch_denoise = st.checkbox(t('segmentation.denoise'), value=False, key="batch_denoise")
            batch_enhance = st.checkbox(t('segmentation.enhance'), value=False, key="batch_enhance")
            batch_normalize = st.checkbox(t('segmentation.normalize'), value=False, key="batch_normalize")

        # 后处理选项
        with st.expander(t('segmentation.postprocessing')):
            batch_closing = st.checkbox(t('segmentation.region_closing'), value=True, key="batch_closing", help=t('segmentation.region_closing_help'))
            if batch_closing:
                batch_closing_kernel_size = st.slider(t('segmentation.closing_kernel_size'), 3, 15, 5, 2, key="batch_closing_kernel", help=t('segmentation.closing_kernel_help'))
            else:
                batch_closing_kernel_size = 5

            batch_extract_cells = st.checkbox(t('segmentation.extract_cells'), value=False, key="batch_extract_cells", help=t('segmentation.extract_cells_help'))
            if batch_extract_cells:
                batch_min_cell_area = st.slider(t('segmentation.min_cell_area'), 50, 500, 100, 10, key="batch_min_cell_area", help=t('segmentation.min_cell_area_help'))

            batch_extract_morphology = st.checkbox(t('segmentation.extract_features'), value=False, key="batch_extract_morphology", help=t('segmentation.extract_features_help'))

            # 批量高级特征提取选项
            batch_use_advanced_features = st.checkbox(t('segmentation.advanced_features'), value=False, key="batch_use_advanced_features", help=t('segmentation.advanced_features_help'))
            if batch_use_advanced_features:
                st.caption(t('segmentation.advanced_feature_categories'))
                batch_include_hu_moments = st.checkbox(t('segmentation.hu_moments'), value=True, key="batch_hu_moments", help=t('segmentation.hu_moments_help'))
                batch_include_intensity = st.checkbox(t('segmentation.intensity_stats'), value=True, key="batch_intensity", help=t('segmentation.intensity_stats_help'))
                batch_include_texture = st.checkbox(t('segmentation.texture_features'), value=True, key="batch_texture", help=t('segmentation.texture_features_help'))
                batch_include_boundary = st.checkbox(t('segmentation.boundary_complexity'), value=True, key="batch_boundary", help=t('segmentation.boundary_complexity_help'))
                batch_include_advanced_shape = st.checkbox(t('segmentation.advanced_shape_features'), value=True, key="batch_advanced_shape", help=t('segmentation.advanced_shape_features_help'))

        # 分割方法
        batch_method = st.selectbox(
            t('segmentation.method'),
            [t('methods.otsu'), t('methods.adaptive'), t('methods.watershed'), t('methods.canny'),
             t('methods.cellpose'), t('methods.cellvit'), t('methods.cellsam')],
            key="batch_method"
        )

        # 方法参数
        with st.expander(t('segmentation.method_params')):
            if batch_method == t('methods.adaptive'):
                batch_block_size = st.slider(t('segmentation.block_size'), 3, 51, 11, 2, key="batch_block_size")
                batch_C = st.slider(t('segmentation.constant_c'), 0, 20, 2, key="batch_C")
                batch_params = {"block_size": batch_block_size, "C": batch_C}
            elif batch_method == t('methods.canny'):
                batch_low = st.slider(t('segmentation.low_threshold'), 0, 200, 50, 10, key="batch_low")
                batch_high = st.slider(t('segmentation.high_threshold'), 0, 300, 150, 10, key="batch_high")
                batch_params = {"low_threshold": batch_low, "high_threshold": batch_high}
            elif batch_method == t('methods.cellpose'):
                batch_model_type = st.selectbox(t('segmentation.model_type'), ["cyto2", "cyto", "nuclei"], key="batch_model_type",
                                               help=t('segmentation.model_type_help'))
                batch_diameter = st.slider(t('segmentation.cell_diameter'), 0, 100, 30, 5, key="batch_diameter",
                                          help=t('segmentation.cell_diameter_help'))
                if batch_diameter == 0:
                    batch_diameter = None

                # GPU选项
                if GPU_AVAILABLE and GPU_COMPATIBLE:
                    batch_use_gpu = st.checkbox(t('segmentation.use_gpu', gpu_name=GPU_NAME), value=True, key="batch_use_gpu",
                                               help=t('segmentation.use_gpu_help'))
                elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                    batch_use_gpu = False
                    st.error(t('segmentation.gpu_incompatible', warning=GPU_WARNING))
                    st.info(t('messages.gpu_cpu_mode'))
                else:
                    batch_use_gpu = False
                    st.info(t('messages.gpu_unavailable'))

                batch_params = {"model_type": batch_model_type, "diameter": batch_diameter, "use_gpu": batch_use_gpu}
            elif batch_method == t('methods.cellvit'):
                # 环境检查
                if not CELLVIT_ENV_OK:
                    st.error(t('messages.cellvit_env_not_found'))
                    st.warning(t('messages.cellvit_env_required'))
                else:
                    st.success(t('messages.cellvit_env_ready', env=CELLVIT_ENV_NAME))

                batch_model_type = st.selectbox(t('segmentation.model_type'), ["CellViT-256"], key="batch_cellvit_model",
                                               help=t('segmentation.model_type_help'))
                batch_target_size = st.slider(t('segmentation.target_size'), 256, 1024, 512, 64, key="batch_target_size",
                                             help=t('segmentation.target_size_help'))

                # GPU选项
                if GPU_AVAILABLE and GPU_COMPATIBLE:
                    batch_use_gpu = st.checkbox(t('segmentation.use_gpu', gpu_name=GPU_NAME), value=True, key="batch_cellvit_gpu",
                                               help=t('segmentation.cellvit_gpu_help'))
                elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                    batch_use_gpu = False
                    st.error(t('segmentation.gpu_incompatible', warning=GPU_WARNING))
                else:
                    batch_use_gpu = False
                    st.info(t('messages.gpu_unavailable'))

                batch_params = {"model_type": batch_model_type, "target_size": batch_target_size, "use_gpu": batch_use_gpu}
            elif batch_method == t('methods.cellsam'):
                # 提示信息
                st.info(t('messages.cellsam_info'))

                batch_model_type = st.selectbox(t('segmentation.model_type'), ["vit_b", "vit_l", "vit_h"], key="batch_cellsam_model",
                                               help=t('segmentation.model_size'))
                batch_points_per_side = st.slider(t('segmentation.points_per_side'), 16, 64, 32, 8, key="batch_points_per_side",
                                                 help=t('segmentation.points_per_side_help'))

                # GPU选项
                if GPU_AVAILABLE and GPU_COMPATIBLE:
                    batch_use_gpu = st.checkbox(t('segmentation.use_gpu', gpu_name=GPU_NAME), value=True, key="batch_cellsam_gpu",
                                               help=t('segmentation.cellsam_gpu_help'))
                elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                    batch_use_gpu = False
                    st.error(t('segmentation.gpu_incompatible', warning=GPU_WARNING))
                else:
                    batch_use_gpu = False
                    st.info(t('messages.gpu_unavailable'))

                batch_params = {"model_type": batch_model_type, "points_per_side": batch_points_per_side, "use_gpu": batch_use_gpu}
            else:
                batch_params = {}

        batch_process_btn = st.button(t('common.batch_process'), type="primary", use_container_width=True)

    with col_batch_right:
        if uploaded_files:
            st.info(t('messages.uploaded_files_count', count=len(uploaded_files)))

            if batch_process_btn:
                st.subheader(t('common.processing_progress'))

                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()

                batch_results = []
                batch_preprocess_options = {
                    'denoise': batch_denoise,
                    'enhance': batch_enhance,
                    'normalize': batch_normalize
                }

                batch_postprocess_options = {
                    'closing': batch_closing,
                    'closing_kernel_size': batch_closing_kernel_size,
                    'extract_cells': batch_extract_cells,
                    'min_cell_area': batch_min_cell_area if batch_extract_cells else 100,
                    'extract_morphology': batch_extract_morphology,
                    'use_advanced_features': batch_use_advanced_features,
                    'include_hu_moments': batch_include_hu_moments if batch_use_advanced_features else False,
                    'include_intensity': batch_include_intensity if batch_use_advanced_features else False,
                    'include_texture': batch_include_texture if batch_use_advanced_features else False,
                    'include_boundary': batch_include_boundary if batch_use_advanced_features else False,
                    'include_advanced_shape': batch_include_advanced_shape if batch_use_advanced_features else False
                }

                # 并行批量处理
                # 准备所有图像数据和参数
                task_args = []
                for uploaded_file in uploaded_files:
                    try:
                        image = Image.open(uploaded_file)
                        task_args.append((
                            image,  # PIL Image对象
                            uploaded_file.name,
                            batch_method,
                            batch_params,
                            batch_preprocess_options,
                            batch_postprocess_options
                        ))
                    except Exception as e:
                        st.warning(t('messages.file_read_failed', filename=uploaded_file.name, error=str(e)))

                # 使用多进程并行处理
                cpu_count = multiprocessing.cpu_count()
                max_workers = max(1, cpu_count // 2)  # 使用一半的CPU核心

                if batch_method == "Cellpose深度学习":
                    status_text.text(f"🧠 使用 {max_workers} 个进程并行处理 {len(task_args)} 张图像（深度学习模型）...")
                else:
                    status_text.text(f"⚙️ 使用 {max_workers} 个进程并行处理 {len(task_args)} 张图像...")

                completed_count = 0
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    future_to_filename = {executor.submit(process_single_image_worker, args): args[1]
                                         for args in task_args}

                    # 收集结果
                    for future in as_completed(future_to_filename):
                        completed_count += 1
                        filename = future_to_filename[future]

                        try:
                            result_dict = future.result()

                            if result_dict['success']:
                                batch_results.append({
                                    'filename': result_dict['filename'],
                                    'result': result_dict['result'],
                                    'image': result_dict['image']
                                })
                            else:
                                st.warning(t('messages.image_processing_failed', filename=result_dict['filename'], error=result_dict['error']))

                        except Exception as e:
                            st.warning(t('messages.image_processing_exception', filename=filename, error=str(e)))

                        # 更新进度
                        progress_bar.progress(completed_count / len(task_args))
                        status_text.text(t('messages.batch_progress', completed=completed_count, total=len(task_args)))

                status_text.text(t('messages.batch_completed'))
                st.session_state.batch_results = batch_results

                # 显示统计摘要
                st.subheader(t('common.statistics_summary'))

                if batch_results:
                    df_data = []
                    for item in batch_results:
                        df_data.append({
                            t('common.filename'): item['filename'],
                            t('metrics.foreground_pixels'): item['result']['foreground_pixels'],
                            t('metrics.foreground_ratio_percent'): f"{item['result']['foreground_ratio']:.2f}",
                            t('metrics.processing_time_ms'): f"{item['result']['processing_time']*1000:.2f}"
                        })

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)

                    # 可视化结果
                    st.subheader(t('common.visualization_results'))

                    # 使用expander来展示每张图片的结果
                    for idx, item in enumerate(batch_results):
                        with st.expander(f"📷 {item['filename']}", expanded=(idx == 0)):
                            col_vis1, col_vis2, col_vis3 = st.columns(3)

                            with col_vis1:
                                st.write(t('common.original_image_bold'))
                                st.image(item['image'], use_container_width=True)

                            with col_vis2:
                                st.write(t('common.segmentation_mask_bold'))
                                st.image(item['result']['mask'], use_container_width=True)

                            with col_vis3:
                                st.write(t('common.overlay_display_bold'))
                                st.image(item['result']['overlay'], use_container_width=True)

                            # 显示统计信息
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric(t('metrics.foreground_pixels'), f"{item['result']['foreground_pixels']:,}")
                            with col_stat2:
                                st.metric(t('metrics.foreground_ratio'), f"{item['result']['foreground_ratio']:.2f}%")
                            with col_stat3:
                                st.metric(t('metrics.processing_time'), f"{item['result']['processing_time']*1000:.2f} ms")

                    # 批量导出
                    st.subheader(t('common.batch_export'))

                    col_export_all1, col_export_all2, col_export_all3 = st.columns(3)

                    # 准备ZIP文件（所有掩码）
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for item in batch_results:
                            mask_bytes = cv2.imencode('.png', item['result']['mask'])[1].tobytes()
                            filename = f"mask_{Path(item['filename']).stem}.png"
                            zip_file.writestr(filename, mask_bytes)

                    # 准备ZIP文件（所有叠加图）
                    overlay_zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(overlay_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for item in batch_results:
                            overlay_bytes = cv2.imencode('.png', item['result']['overlay'])[1].tobytes()
                            filename = f"overlay_{Path(item['filename']).stem}.png"
                            zip_file.writestr(filename, overlay_bytes)

                    # 准备CSV文件（使用GBK编码避免乱码）
                    csv_buffer = io.BytesIO()
                    df.to_csv(csv_buffer, index=False, encoding='gbk')

                    with col_export_all1:
                        st.download_button(
                            t('common.export_all_masks_zip'),
                            data=zip_buffer.getvalue(),
                            file_name=f"masks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

                    with col_export_all2:
                        st.download_button(
                            t('common.export_all_overlays_zip'),
                            data=overlay_zip_buffer.getvalue(),
                            file_name=f"overlays_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

                    with col_export_all3:
                        st.download_button(
                            t('common.export_statistics_report_csv'),
                            data=csv_buffer.getvalue(),
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    # 单个细胞批量导出
                    total_cells = sum(len(item['result']['individual_cells']) if item['result']['individual_cells'] else 0 for item in batch_results)
                    if total_cells > 0:
                        st.write("")  # 添加间距
                        st.info(t('messages.total_cells_extracted', count=total_cells))

                        # 准备所有细胞的ZIP文件
                        all_cells_zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(all_cells_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for item in batch_results:
                                if item['result']['individual_cells']:
                                    img_name = Path(item['filename']).stem
                                    for idx, cell_data in enumerate(item['result']['individual_cells']):
                                        # 保存细胞图像
                                        cell_img_bytes = cv2.imencode('.png', cell_data['image'])[1].tobytes()
                                        zip_file.writestr(f"{img_name}_cell_{idx+1:03d}_image.png", cell_img_bytes)

                                        # 保存细胞掩码
                                        cell_mask_bytes = cv2.imencode('.png', cell_data['mask'])[1].tobytes()
                                        zip_file.writestr(f"{img_name}_cell_{idx+1:03d}_mask.png", cell_mask_bytes)

                        st.download_button(
                            t('common.export_all_cell_samples_zip'),
                            data=all_cells_zip_buffer.getvalue(),
                            file_name=f"all_cells_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

        else:
            st.info(t('messages.upload_multiple_images_left'))

# ==================== 标签页6: 异常检测 ====================
with tab6:
    st.header(t('tabs.anomaly_detection'))
    st.caption(t('help.anomaly_detection_caption'))
    st.info(t('messages.usage_instructions_see_top'))

    st.markdown("---")

    # CSV文件上传
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded_csv = st.file_uploader(
            t('common.upload_csv_file'),
            type=["csv"],
            key="ml_csv_upload",
            help=t('help.upload_csv_help')
        )

    with col_info:
        if uploaded_csv is not None:
            st.success(t('messages.file_uploaded'))
            st.info(t('messages.filename_info', name=uploaded_csv.name))

    # 处理上传的CSV文件
    if uploaded_csv is not None:
        try:
            # 读取CSV文件
            ml_features_df = pd.read_csv(uploaded_csv)

            # 验证CSV格式
            required_cols = ['area_um2', 'perimeter_um', 'circularity']
            missing_cols = [col for col in required_cols if col not in ml_features_df.columns]

            if missing_cols:
                st.error(t('messages.csv_format_incorrect', cols=', '.join(missing_cols)))
                st.info(t('messages.csv_should_contain_features'))
            else:
                # 显示数据概览
                st.success(t('messages.csv_loaded_successfully', count=len(ml_features_df)))

                # 数据预览
                with st.expander(t('common.data_preview'), expanded=False):
                    st.write(t('common.data_dimensions', rows=ml_features_df.shape[0], cols=ml_features_df.shape[1]))
                    st.write(t('common.first_10_rows'))
                    st.dataframe(ml_features_df.head(10), use_container_width=True)

                    # 显示特征列表
                    feature_cols = [col for col in ml_features_df.columns if col not in
                                   ['sequential_id', 'cell_id', 'centroid_x', 'centroid_y',
                                    'bbox_min_row', 'bbox_min_col', 'bbox_max_row', 'bbox_max_col']]
                    st.write(t('common.available_features', count=len(feature_cols)))
                    st.write(", ".join(feature_cols))

                st.markdown("---")

                # 保存到session_state
                st.session_state['ml_features_df'] = ml_features_df

        except Exception as e:
            st.error(t('messages.csv_read_failed', error=str(e)))
            st.info(t('messages.ensure_valid_csv'))

    # 机器学习异常识别UI（只在有数据时显示）
    if 'ml_features_df' in st.session_state and not st.session_state['ml_features_df'].empty:
        ml_features_df = st.session_state['ml_features_df']

        st.markdown("---")

        st.subheader(t('ml.anomaly_detection_title'))
        st.caption(t('ml.anomaly_detection_caption'))

        # 异常检测算法选择
        col_anomaly_algo, col_anomaly_param = st.columns([1, 2])

        with col_anomaly_algo:
            ml_anomaly_method = st.selectbox(
                t('ml.anomaly_detection_algorithm'),
                ["Isolation Forest", "LOF", "One-Class SVM", "Elliptic Envelope"],
                key="ml_anomaly_method",
                help=t('ml.anomaly_algorithm_help')
            )

        with col_anomaly_param:
            # 根据选择的算法显示不同的参数控件
            if ml_anomaly_method == "Isolation Forest":
                ml_contamination = st.slider(
                    t('ml.contamination_ratio'),
                    0.01, 0.5, 0.1, 0.01,
                    key="ml_contamination_if",
                    help=t('ml.contamination_help')
                )

            elif ml_anomaly_method == "LOF":
                col_cont, col_neigh = st.columns(2)
                with col_cont:
                    ml_contamination = st.slider(
                        t('ml.contamination_ratio'),
                        0.01, 0.5, 0.1, 0.01,
                        key="ml_contamination_lof"
                    )
                with col_neigh:
                    ml_n_neighbors_lof = st.slider(
                        t('ml.n_neighbors'),
                        5, 50, 20, 5,
                        key="ml_n_neighbors_lof",
                        help=t('ml.n_neighbors_help')
                    )

            elif ml_anomaly_method == "One-Class SVM":
                col_nu, col_kernel = st.columns(2)
                with col_nu:
                    ml_nu = st.slider(
                        t('ml.nu_upper_bound'),
                        0.01, 0.5, 0.1, 0.01,
                        key="ml_nu",
                        help=t('ml.nu_upper_bound_help')
                    )
                with col_kernel:
                    ml_kernel = st.selectbox(
                        t('ml.kernel_function'),
                        ["rbf", "linear", "poly", "sigmoid"],
                        key="ml_kernel"
                    )

            elif ml_anomaly_method == "Elliptic Envelope":
                ml_contamination = st.slider(
                    t('ml.contamination_ratio'),
                    0.01, 0.5, 0.1, 0.01,
                    key="ml_contamination_ee",
                    help=t('ml.contamination_help')
                )

        # 执行异常检测按钮
        if st.button(t('ml.execute_anomaly_detection'), type="primary", use_container_width=True, key="ml_anomaly_button"):
            with st.spinner(t('ml.executing_anomaly_detection')):
                try:
                    # 执行异常检测
                    if ml_anomaly_method == "Isolation Forest":
                        labels, info = detect_isolation_forest(ml_features_df, contamination=ml_contamination)

                    elif ml_anomaly_method == "LOF":
                        labels, info = detect_lof(ml_features_df, contamination=ml_contamination,
                                                 n_neighbors=ml_n_neighbors_lof)

                    elif ml_anomaly_method == "One-Class SVM":
                        labels, info = detect_one_class_svm(ml_features_df, nu=ml_nu, kernel=ml_kernel)

                    elif ml_anomaly_method == "Elliptic Envelope":
                        labels, info = detect_elliptic_envelope(ml_features_df, contamination=ml_contamination)

                    # 保存异常检测结果到session_state
                    st.session_state['ml_anomaly_labels'] = labels
                    st.session_state['ml_anomaly_info'] = info

                    # 将异常标签添加到features_df
                    ml_features_df_anomaly = ml_features_df.copy()
                    ml_features_df_anomaly['anomaly'] = labels
                    ml_features_df_anomaly['anomaly_label'] = ml_features_df_anomaly['anomaly'].map({1: '正常', -1: '异常'})
                    st.session_state['ml_features_df_anomaly'] = ml_features_df_anomaly

                    st.success(t('ml.anomaly_detection_completed'))

                    # 显示异常检测结果
                    st.write(t('ml.anomaly_detection_results'))
                    col_r1, col_r2, col_r3 = st.columns(3)

                    with col_r1:
                        st.metric(t('ml.normal_samples'), f"{info['n_normal']} 个",
                                 help=t('ml.normal_samples_help'))
                    with col_r2:
                        st.metric(t('ml.anomaly_samples'), f"{info['n_anomalies']} 个",
                                 help=t('ml.anomaly_samples_help'))
                    with col_r3:
                        st.metric(t('ml.anomaly_ratio'), f"{info['anomaly_ratio']*100:.1f}%",
                                 help=t('ml.anomaly_ratio_help'))

                except Exception as e:
                    st.error(t('ml.anomaly_detection_failed', error=str(e)))

        # 异常检测可视化和统计（只在有异常检测结果时显示）
        if 'ml_anomaly_labels' in st.session_state and 'ml_features_df' in st.session_state:
            st.markdown("---")
            with st.expander(t('ml.anomaly_visualization_stats'), expanded=True):
                st.caption(t('ml.visualize_anomaly_distribution'))

                # 异常统计分析
                st.write(t('ml.anomaly_feature_statistics'))
                st.caption(t('ml.compare_normal_anomaly_features'))

                try:
                    ml_anomaly_labels = st.session_state['ml_anomaly_labels']
                    ml_features_df = st.session_state['ml_features_df']

                    # 获取异常统计
                    anomaly_stats = get_anomaly_statistics(ml_features_df, ml_anomaly_labels)

                    # 显示前10个差异最大的特征
                    st.dataframe(
                        anomaly_stats.head(10),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(t('ml.difference_column_explanation'))

                except Exception as e:
                    st.error(t('ml.statistical_analysis_failed', error=str(e)))

            st.markdown("---")

            # 算法可视化
            st.write(t('ml.algorithm_visualization'))
            st.caption(t('ml.show_algorithm_principle'))

            try:
                ml_anomaly_info = st.session_state['ml_anomaly_info']
                ml_anomaly_labels = st.session_state['ml_anomaly_labels']
                ml_features_df = st.session_state['ml_features_df']

                # 获取算法信息
                method = ml_anomaly_info['method']
                scores = ml_anomaly_info['scores']
                scaler = ml_anomaly_info['scaler']
                feature_cols = ml_anomaly_info['feature_cols']

                # 预处理特征（使用保存的scaler）
                features = ml_features_df[feature_cols].values
                features_scaled = scaler.transform(features)

                # 根据算法类型调用相应的可视化函数
                if method == "Isolation Forest":
                    fig = visualize_isolation_forest(features_scaled, ml_anomaly_labels, scores, ml_anomaly_info)
                elif method == "Local Outlier Factor":
                    fig = visualize_lof(features_scaled, ml_anomaly_labels, scores, ml_anomaly_info)
                elif method == "One-Class SVM":
                    fig = visualize_one_class_svm(features_scaled, ml_anomaly_labels, scores, ml_anomaly_info)
                elif method == "Elliptic Envelope":
                    fig = visualize_elliptic_envelope(features_scaled, ml_anomaly_labels, scores, ml_anomaly_info)

                # 显示可视化 - 根据数据量智能选择格式
                n_samples = len(ml_features_df)
                if n_samples < 10000:
                    # 数据点少于10000，使用SVG矢量图（超高清，可无限放大）
                    svg_buffer = io.BytesIO()
                    fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
                    svg_buffer.seek(0)
                    svg_str = svg_buffer.getvalue().decode('utf-8')
                    st.components.v1.html(svg_str, height=900, scrolling=True)
                    st.caption(t('ml.visualization_layout_caption'))
                    st.info(t('ml.svg_vector_display', n_samples=n_samples))
                else:
                    # 数据点多于10000，使用PNG栅格图（避免浏览器卡顿）
                    st.pyplot(fig)
                    st.caption(t('ml.visualization_layout_caption'))
                    st.info(t('ml.png_raster_display', n_samples=n_samples))

                # 下载按钮
                st.write(t('ml.download_visualization'))
                col_dl1, col_dl2, col_dl3 = st.columns(3)

                with col_dl1:
                    # PNG格式
                    png_buffer = io.BytesIO()
                    fig.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
                    png_buffer.seek(0)
                    st.download_button(
                        t('ml.download_png'),
                        data=png_buffer,
                        file_name=f"anomaly_{method.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with col_dl2:
                    # SVG格式
                    svg_buffer = io.BytesIO()
                    fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
                    svg_buffer.seek(0)
                    st.download_button(
                        t('ml.download_svg'),
                        data=svg_buffer,
                        file_name=f"anomaly_{method.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                        mime="image/svg+xml",
                        use_container_width=True
                    )

                with col_dl3:
                    # PDF格式
                    pdf_buffer = io.BytesIO()
                    fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight')
                    pdf_buffer.seek(0)
                    st.download_button(
                        t('ml.download_pdf'),
                        data=pdf_buffer,
                        file_name=f"anomaly_{method.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(t('ml.algorithm_visualization_failed', error=str(e)))

            st.markdown("---")

            # 降维可视化
            st.write(t('ml.anomaly_sample_visualization'))
            col_viz_method_anomaly, col_viz_param_anomaly = st.columns([1, 2])

            with col_viz_method_anomaly:
                ml_viz_method_anomaly = st.selectbox(
                    t('ml.dimensionality_reduction_method'),
                    ["PCA", "t-SNE", "UMAP"],
                    key="ml_viz_method_anomaly",
                    help=t('ml.dimensionality_reduction_help')
                )

            with col_viz_param_anomaly:
                if ml_viz_method_anomaly == "t-SNE":
                    ml_perplexity_anomaly = st.slider(
                        t('ml.perplexity'),
                        5, 50, 30, 5,
                        key="ml_perplexity_anomaly"
                    )
                elif ml_viz_method_anomaly == "UMAP":
                    col_n, col_d = st.columns(2)
                    with col_n:
                        ml_n_neighbors_anomaly = st.slider(
                            t('ml.n_neighbors'),
                            5, 50, 15, 5,
                            key="ml_n_neighbors_anomaly"
                        )
                    with col_d:
                        ml_min_dist_anomaly = st.slider(
                            t('ml.min_distance'),
                            0.0, 0.99, 0.1, 0.05,
                            key="ml_min_dist_anomaly"
                        )

            # 执行可视化
            if st.button(t('ml.generate_anomaly_visualization'), type="secondary", use_container_width=True, key="ml_viz_anomaly_button"):
                with st.spinner(t('ml.executing_dimensionality_reduction', method=ml_viz_method_anomaly)):
                    try:
                        ml_features_df = st.session_state['ml_features_df']
                        ml_anomaly_labels = st.session_state['ml_anomaly_labels']

                        # 执行降维
                        if ml_viz_method_anomaly == "PCA":
                            components, _, _ = apply_pca(ml_features_df, n_components=2)
                        elif ml_viz_method_anomaly == "t-SNE":
                            components, _ = apply_tsne(ml_features_df, n_components=2,
                                                      perplexity=ml_perplexity_anomaly, n_iter=1000)
                        elif ml_viz_method_anomaly == "UMAP":
                            components, _ = apply_umap(ml_features_df, n_components=2,
                                                      n_neighbors=ml_n_neighbors_anomaly,
                                                      min_dist=ml_min_dist_anomaly)

                        # 创建可视化DataFrame
                        viz_df_anomaly = pd.DataFrame({
                            'component_1': components[:, 0],
                            'component_2': components[:, 1],
                            'anomaly_label': ml_anomaly_labels
                        })
                        viz_df_anomaly['status'] = viz_df_anomaly['anomaly_label'].map({1: t('ml.normal'), -1: t('ml.anomaly')})

                        # 添加特征用于hover
                        if 'sequential_id' in ml_features_df.columns:
                            viz_df_anomaly['sequential_id'] = ml_features_df['sequential_id'].values
                        if 'area_um2' in ml_features_df.columns:
                            viz_df_anomaly['area_um2'] = ml_features_df['area_um2'].values
                        if 'circularity' in ml_features_df.columns:
                            viz_df_anomaly['circularity'] = ml_features_df['circularity'].values

                        # 创建交互式散点图
                        fig = px.scatter(
                            viz_df_anomaly,
                            x='component_1',
                            y='component_2',
                            color='status',
                            hover_data=['sequential_id', 'area_um2', 'circularity'] if 'sequential_id' in viz_df_anomaly.columns else None,
                            title=t('ml.anomaly_detection_visualization_title', method=ml_viz_method_anomaly),
                            labels={'component_1': f'{ml_viz_method_anomaly}1',
                                   'component_2': f'{ml_viz_method_anomaly}2',
                                   'status': t('ml.sample_status')},
                            color_discrete_map={t('ml.normal'): '#2ecc71', t('ml.anomaly'): '#e74c3c'}
                        )

                        fig.update_traces(marker=dict(size=8, opacity=0.7))
                        fig.update_layout(height=600)

                        st.plotly_chart(fig, use_container_width=True)

                        st.success(t('ml.visualization_completed', method=ml_viz_method_anomaly))

                        # 下载按钮
                        st.write(t('ml.download_visualization'))
                        col_dl_plotly1, col_dl_plotly2, col_dl_plotly3 = st.columns(3)

                        with col_dl_plotly1:
                            # HTML格式（交互式）
                            html_buffer = io.StringIO()
                            fig.write_html(html_buffer)
                            html_str = html_buffer.getvalue()
                            st.download_button(
                                t('ml.download_html_interactive'),
                                data=html_str,
                                file_name=f"anomaly_viz_{ml_viz_method_anomaly}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html",
                                use_container_width=True,
                                help=t('ml.download_html_help')
                            )

                        with col_dl_plotly2:
                            # PNG格式
                            try:
                                png_bytes = fig.to_image(format="png", width=1200, height=800)
                                st.download_button(
                                    t('ml.download_png'),
                                    data=png_bytes,
                                    file_name=f"anomaly_viz_{ml_viz_method_anomaly}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.caption(t('ml.kaleido_required_png'))

                        with col_dl_plotly3:
                            # SVG格式
                            try:
                                svg_bytes = fig.to_image(format="svg", width=1200, height=800)
                                st.download_button(
                                    t('ml.download_svg'),
                                    data=svg_bytes,
                                    file_name=f"anomaly_viz_{ml_viz_method_anomaly}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                                    mime="image/svg+xml",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.caption(t('ml.kaleido_required_svg'))

                    except Exception as e:
                        st.error(t('ml.visualization_failed', error=str(e)))

    # 结果导出section（只在有聚类或异常检测结果时显示）
    if 'ml_features_df_clustered' in st.session_state or 'ml_features_df_anomaly' in st.session_state:
        st.markdown("---")
        st.subheader(t('ml.export_results'))

        # 聚类结果导出
        if 'ml_features_df_clustered' in st.session_state:
            ml_features_df_clustered = st.session_state['ml_features_df_clustered']

            col_export1, col_export2 = st.columns(2)

            with col_export1:
                # 导出带聚类标签的CSV
                csv_data = ml_features_df_clustered.to_csv(index=False)
                st.download_button(
                    t('ml.download_clustered_csv'),
                    data=csv_data,
                    file_name=f"clustered_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help=t('ml.download_clustered_csv_help')
                )

            with col_export2:
                # 显示数据预览
                st.info(t('ml.clustered_data_info', cells=len(ml_features_df_clustered), features=len(ml_features_df_clustered.columns)))

        # 异常检测结果导出
        if 'ml_features_df_anomaly' in st.session_state:
            ml_features_df_anomaly = st.session_state['ml_features_df_anomaly']

            col_export3, col_export4 = st.columns(2)

            with col_export3:
                # 导出带异常标签的CSV
                csv_data_anomaly = ml_features_df_anomaly.to_csv(index=False)
                st.download_button(
                    t('ml.download_anomaly_csv'),
                    data=csv_data_anomaly,
                    file_name=f"anomaly_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help=t('ml.download_anomaly_csv_help')
                )

            with col_export4:
                # 显示数据预览
                n_normal = (ml_features_df_anomaly['anomaly'] == 1).sum()
                n_anomaly = (ml_features_df_anomaly['anomaly'] == -1).sum()
                st.info(t('ml.anomaly_data_info', cells=len(ml_features_df_anomaly), normal=n_normal, anomaly=n_anomaly))

            # 导出仅正常样本的CSV
            st.write("")
            col_export5, col_export6 = st.columns(2)

            with col_export5:
                # 导出仅正常样本
                ml_features_df_normal = ml_features_df_anomaly[ml_features_df_anomaly['anomaly'] == 1].copy()
                csv_data_normal = ml_features_df_normal.to_csv(index=False)
                st.download_button(
                    t('ml.download_normal_only_csv'),
                    data=csv_data_normal,
                    file_name=f"normal_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help=t('ml.download_normal_only_csv_help')
                )

            with col_export6:
                st.caption(t('ml.normal_samples_info', cells=len(ml_features_df_normal), anomaly=n_anomaly))

    else:
        st.info(t('common.please_upload_above'))

# ==================== 标签页7: 聚类分析 ====================
with tab7:
    st.header(t('ml.clustering_analysis_title'))
    st.caption(t('ml.clustering_analysis_caption'))
    st.info(t('messages.usage_instructions_see_top'))

    st.markdown("---")

    # CSV文件上传
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded_csv = st.file_uploader(
            t('ml.upload_clustering_csv'),
            type=['csv'],
            key="clustering_csv_upload",
            help=t('ml.upload_clustering_csv_help')
        )

    with col_info:
        st.info(t('ml.csv_should_contain_morphology'))

    # 读取CSV文件
    if uploaded_csv is not None:
        try:
            import pandas as pd
            clustering_features_df = pd.read_csv(uploaded_csv)

            st.success(t('ml.csv_read_success_clustering', count=len(clustering_features_df)))

            # 显示数据预览
            with st.expander(t('ml.data_preview'), expanded=False):
                st.write(t('ml.data_dimensions', rows=clustering_features_df.shape[0], cols=clustering_features_df.shape[1]))
                st.dataframe(clustering_features_df.head(10), use_container_width=True)

                # 显示特征列表
                st.write(t('ml.feature_list'))
                feature_cols = [col for col in clustering_features_df.columns if col not in ['cell_id', 'sequential_id', 'image_name']]
                st.write(", ".join(feature_cols))

            # 保存到session_state
            st.session_state['clustering_features_df'] = clustering_features_df

        except Exception as e:
            st.error(t('ml.csv_read_failed', error=str(e)))
            st.info(t('ml.ensure_valid_csv'))

    # 聚类分析UI（只在有数据时显示）
    if 'clustering_features_df' in st.session_state and not st.session_state['clustering_features_df'].empty:
        clustering_features_df = st.session_state['clustering_features_df']

        st.markdown("---")
        st.subheader(t('ml.clustering_analysis_subheader'))

        # Left-right column layout for clustering
        col_left, col_right = st.columns([2, 3])

        with col_left:
            st.write(t('ml.clustering_parameters'))

            # Clustering algorithm selection
            clustering_method = st.selectbox(
                t('ml.clustering_algorithm'),
                ["kmeans", "dbscan", "hierarchical", "gmm"],
                format_func=lambda x: {
                    "kmeans": "K-means",
                    "dbscan": "DBSCAN",
                    "hierarchical": t('ml.hierarchical_clustering'),
                    "gmm": "GMM"
                }[x],
                key="clustering_method",
                help=t('ml.clustering_algorithm_help')
            )

            # Algorithm-specific parameters
            if clustering_method == "kmeans":
                col_k, col_auto = st.columns(2)
                with col_k:
                    n_clusters = st.number_input(t('ml.n_clusters'), min_value=2, max_value=10, value=3, step=1, key="clustering_n_clusters")
                with col_auto:
                    auto_k = st.checkbox(t('ml.auto_detect_k'), value=False, key="clustering_auto_k", help=t('ml.auto_detect_k_help'))

            elif clustering_method == "dbscan":
                col_eps, col_min = st.columns(2)
                with col_eps:
                    eps = st.number_input(t('ml.eps_radius'), min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="clustering_eps")
                    auto_eps = st.checkbox(t('ml.auto_estimate_eps'), value=True, key="clustering_auto_eps")
                with col_min:
                    min_samples = st.number_input(t('ml.min_samples'), min_value=2, max_value=20, value=5, step=1, key="clustering_min_samples")

            elif clustering_method == "hierarchical":
                col_k, col_link = st.columns(2)
                with col_k:
                    n_clusters = st.number_input(t('ml.n_clusters'), min_value=2, max_value=10, value=3, step=1, key="clustering_n_clusters_hier")
                with col_link:
                    linkage = st.selectbox(t('ml.linkage_method'), ["ward", "complete", "average", "single"], key="clustering_linkage")

            elif clustering_method == "gmm":
                n_components = st.number_input(t('ml.n_components_gmm'), min_value=2, max_value=10, value=3, step=1, key="clustering_n_components")

            # Execute clustering button
            if st.button(t('ml.execute_clustering'), type="primary", use_container_width=True, key="clustering_execute_button"):
                with st.spinner(t('ml.executing_clustering')):
                    try:
                        # Execute clustering
                        if clustering_method == "kmeans":
                            if auto_k:
                                optimal_results = find_optimal_clusters(clustering_features_df, max_k=10, method='kmeans')
                                n_clusters = optimal_results['best_k']
                                st.info(t('ml.auto_detected_optimal_k', n_clusters=n_clusters))
                            labels, info = perform_kmeans(clustering_features_df, n_clusters=n_clusters)

                        elif clustering_method == "dbscan":
                            eps_val = None if auto_eps else eps
                            labels, info = perform_dbscan(clustering_features_df, eps=eps_val, min_samples=min_samples)

                        elif clustering_method == "hierarchical":
                            labels, info = perform_hierarchical(clustering_features_df, n_clusters=n_clusters, linkage=linkage)

                        elif clustering_method == "gmm":
                            labels, info = perform_gmm(clustering_features_df, n_components=n_components)

                        # Save clustering results to session_state
                        st.session_state['clustering_labels'] = labels
                        st.session_state['clustering_info'] = info

                        # Add clustering labels to features_df
                        clustering_features_df_clustered = clustering_features_df.copy()
                        clustering_features_df_clustered['cluster'] = labels
                        st.session_state['clustering_features_df_clustered'] = clustering_features_df_clustered

                        st.success(t('ml.clustering_completed'))

                    except Exception as e:
                        st.error(t('ml.clustering_failed', error=str(e)))

            # Display clustering results (only when results exist)
            if 'clustering_labels' in st.session_state and 'clustering_info' in st.session_state:
                st.markdown("---")
                st.write(t('ml.clustering_quality_metrics'))

                info = st.session_state['clustering_info']
                labels = st.session_state['clustering_labels']

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric(t('ml.silhouette_score'), f"{info['silhouette_score']:.3f}", help=t('ml.silhouette_score_help'))
                with col_m2:
                    st.metric(t('ml.davies_bouldin_score'), f"{info['davies_bouldin_score']:.3f}", help=t('ml.davies_bouldin_score_help'))

                st.write(t('ml.clustering_statistics'))
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                st.write(t('ml.discovered_clusters', n_clusters=n_clusters_found))

                if -1 in labels:
                    n_noise = list(labels).count(-1)
                    st.write(t('ml.noise_points', n_noise=n_noise))

                cluster_counts = pd.Series(labels).value_counts().sort_index()
                for cluster_id, count in cluster_counts.items():
                    if cluster_id != -1:
                        st.write(t('ml.cluster_cell_count', cluster_id=cluster_id, count=count))

        with col_right:
            st.write(t('ml.clustering_visualization'))

            # Visualization only when clustering results exist
            if 'clustering_labels' in st.session_state and 'clustering_features_df' in st.session_state:

                # Dimensionality reduction method selection
                viz_method = st.selectbox(
                    t('ml.dimensionality_reduction_method'),
                    ["PCA", "t-SNE", "UMAP"],
                    key="clustering_viz_method",
                    help=t('ml.dimensionality_reduction_help')
                )

                # Method-specific parameters
                if viz_method == "t-SNE":
                    col_perp, col_iter = st.columns(2)
                    with col_perp:
                        perplexity = st.slider(t('ml.perplexity'), 5, 50, 30, 5, key="clustering_perplexity")
                    with col_iter:
                        n_iter = st.slider(t('ml.n_iterations'), 250, 2000, 1000, 250, key="clustering_n_iter")
                elif viz_method == "UMAP":
                    col_neighbors, col_dist = st.columns(2)
                    with col_neighbors:
                        n_neighbors = st.slider(t('ml.n_neighbors'), 5, 50, 15, 5, key="clustering_n_neighbors")
                    with col_dist:
                        min_dist = st.slider(t('ml.min_distance'), 0.0, 0.99, 0.1, 0.05, key="clustering_min_dist")

                # Execute visualization
                if st.button(t('ml.generate_visualization'), type="secondary", use_container_width=True, key="clustering_viz_button"):
                    with st.spinner(t('ml.executing_dimensionality_reduction', method=viz_method)):
                        try:
                            clustering_features_df = st.session_state['clustering_features_df']
                            clustering_labels = st.session_state['clustering_labels']

                            # Execute dimensionality reduction
                            if viz_method == "PCA":
                                components, _, _ = apply_pca(clustering_features_df, n_components=2)
                            elif viz_method == "t-SNE":
                                components, _ = apply_tsne(clustering_features_df, n_components=2, perplexity=perplexity, n_iter=n_iter)
                            elif viz_method == "UMAP":
                                components, _ = apply_umap(clustering_features_df, n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)

                            # Create visualization DataFrame
                            viz_df = pd.DataFrame({
                                'component_1': components[:, 0],
                                'component_2': components[:, 1],
                                'cluster': clustering_labels.astype(str)
                            })

                            # Add features for hover display
                            if 'sequential_id' in clustering_features_df.columns:
                                viz_df['sequential_id'] = clustering_features_df['sequential_id'].values
                            if 'area_um2' in clustering_features_df.columns:
                                viz_df['area_um2'] = clustering_features_df['area_um2'].values
                            if 'circularity' in clustering_features_df.columns:
                                viz_df['circularity'] = clustering_features_df['circularity'].values

                            # Create interactive scatter plot
                            fig = px.scatter(
                                viz_df,
                                x='component_1',
                                y='component_2',
                                color='cluster',
                                hover_data=['sequential_id', 'area_um2', 'circularity'] if 'sequential_id' in viz_df.columns else None,
                                title=t('ml.clustering_visualization_title', method=viz_method),
                                labels={'component_1': f'{viz_method}1', 'component_2': f'{viz_method}2', 'cluster': t('ml.cluster_label')},
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )

                            fig.update_traces(marker=dict(size=8, opacity=0.7))
                            fig.update_layout(height=500)

                            # Save to session_state for display outside button block
                            st.session_state['clustering_viz_fig'] = fig

                            st.success(t('ml.clustering_viz_completed', method=viz_method))

                        except Exception as e:
                            st.error(t('ml.clustering_viz_failed', error=str(e)))

                # Display saved visualization
                if 'clustering_viz_fig' in st.session_state:
                    st.plotly_chart(st.session_state['clustering_viz_fig'], use_container_width=True, key="clustering_viz_saved")

                    # Download buttons
                    st.write(t('ml.download_visualization'))
                    col_dl1, col_dl2 = st.columns(2)

                    with col_dl1:
                        html_buffer = io.StringIO()
                        st.session_state['clustering_viz_fig'].write_html(html_buffer)
                        html_str = html_buffer.getvalue()
                        st.download_button(
                            t('ml.download_html'),
                            data=html_str,
                            file_name=f"clustering_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True
                        )

                    with col_dl2:
                        try:
                            png_bytes = st.session_state['clustering_viz_fig'].to_image(format="png", width=1200, height=800)
                            st.download_button(
                                t('ml.download_png'),
                                data=png_bytes,
                                file_name=f"clustering_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        except:
                            st.caption(t('ml.png_export_requires_kaleido'))

            else:
                st.info(t('ml.please_execute_clustering_first'))

        # Export section
        if 'clustering_features_df_clustered' in st.session_state:
            st.markdown("---")
            st.subheader(t('ml.export_results'))

            clustering_features_df_clustered = st.session_state['clustering_features_df_clustered']

            col_export1, col_export2 = st.columns(2)

            with col_export1:
                csv_data = clustering_features_df_clustered.to_csv(index=False)
                st.download_button(
                    t('ml.download_clustered_csv'),
                    data=csv_data,
                    file_name=f"clustering_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help=t('ml.download_clustered_csv_help')
                )

            with col_export2:
                st.info(t('ml.clustered_data_info', cells=len(clustering_features_df_clustered), features=len(clustering_features_df_clustered.columns)))


    else:
        st.info(t('ml.please_upload_csv_clustering'))

# ==================== 标签页5: 细胞形态学提取 ====================
with tab5:
    st.header(t('morphology.header'))

    st.markdown(t('morphology.feature_description'))

    # 模式选择
    morphology_mode = st.radio(
        t('morphology.select_mode'),
        [t('morphology.csv_processing'), t('morphology.direct_extraction')],
        horizontal=True,
        help=t('morphology.select_mode_help')
    )

    # ==================== 模式1: CSV数据处理 ====================
    if morphology_mode == t('morphology.csv_processing'):
        st.subheader(t('morphology.csv_processing'))

        st.info(t('morphology.csv_usage_instructions'))

        # CSV文件上传
        uploaded_csv = st.file_uploader(
            t('morphology.upload_feature_csv'),
            type=["csv"],
            key="morphology_csv_upload",
            help=t('morphology.upload_feature_csv_help')
        )

        if uploaded_csv is not None:
            try:
                # 读取CSV文件
                features_df = pd.read_csv(uploaded_csv)

                st.success(t('morphology.csv_loaded_successfully', count=len(features_df)))

                # 显示数据预览
                with st.expander(t('morphology.data_preview_title'), expanded=False):
                    st.dataframe(features_df.head(10), use_container_width=True)
                    st.caption(t('morphology.showing_first_rows', count=len(features_df)))

                # 特征统计分析
                st.subheader(t('morphology.feature_statistics_summary'))

                # 检查是否包含基础特征列
                basic_features = ['area_um2', 'circularity', 'major_axis_length', 'minor_axis_length']
                has_basic_features = all(col in features_df.columns for col in basic_features)

                if has_basic_features:
                    # 计算统计信息
                    stats = get_feature_statistics(features_df)

                    # 显示关键特征的统计信息
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(t('morphology.average_area'), f"{stats['area_um2']['mean']:.1f} μm²")
                    with col2:
                        st.metric(t('morphology.average_circularity'), f"{stats['circularity']['mean']:.3f}")
                    with col3:
                        st.metric(t('morphology.average_major_axis'), f"{stats['major_axis_length']['mean']:.1f} μm")
                    with col4:
                        st.metric(t('morphology.average_minor_axis'), f"{stats['minor_axis_length']['mean']:.1f} μm")

                    # 显示详细统计表格
                    with st.expander(t('morphology.detailed_statistics'), expanded=False):
                        stats_df = pd.DataFrame(stats).T
                        st.dataframe(stats_df.round(3), use_container_width=True)
                else:
                    st.warning(t('messages.csv_missing_features'))

                # 特征分布可视化
                st.subheader(t('morphology.feature_distribution_visualization'))

                if has_basic_features:
                    # 选择要可视化的特征
                    viz_feature = st.selectbox(
                        t('morphology.select_feature_to_visualize'),
                        options=['area_um2', 'circularity', 'major_axis_length', 'minor_axis_length',
                                'perimeter_um', 'eccentricity', 'solidity', 'aspect_ratio'],
                        format_func=lambda x: t(f'morphology.{x}')
                    )

                    if viz_feature in features_df.columns:
                        # 创建直方图
                        fig = px.histogram(
                            features_df,
                            x=viz_feature,
                            nbins=30,
                            title=t('morphology.feature_distribution', feature=viz_feature),
                            labels={viz_feature: viz_feature}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(t('morphology.feature_not_found', feature=viz_feature))

                # CSV下载功能
                st.subheader(t('morphology.export_data'))

                col_download1, col_download2 = st.columns(2)

                with col_download1:
                    # 下载原始CSV
                    csv_data = features_df.to_csv(index=False)
                    st.download_button(
                        t('morphology.download_csv_file'),
                        data=csv_data,
                        file_name=f"morphology_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help=t('morphology.download_csv_help')
                    )

                with col_download2:
                    if has_basic_features:
                        # 下载统计摘要
                        stats_csv = pd.DataFrame(stats).T.to_csv()
                        st.download_button(
                            t('morphology.download_statistics_summary'),
                            data=stats_csv,
                            file_name=f"morphology_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help=t('morphology.download_statistics_help')
                        )

            except Exception as e:
                st.error(t('morphology.csv_read_error', error=str(e)))
                st.info(t('morphology.ensure_valid_csv_with_features'))

        else:
            st.info(t('morphology.please_upload_csv_to_start'))

    # ==================== 模式2: 直接特征提取 ====================
    elif morphology_mode == t('morphology.direct_extraction'):
        st.subheader(t('morphology.direct_extraction'))

        st.info(t('morphology.direct_extraction_usage'))

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.write(t('morphology.settings'))

            # 图像上传
            uploaded_image = st.file_uploader(
                t('morphology.upload_original_image'),
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="morphology_image_upload",
                help=t('morphology.upload_original_image_help')
            )

            # 掩码上传
            uploaded_mask = st.file_uploader(
                t('morphology.upload_segmentation_mask'),
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="morphology_mask_upload",
                help=t('morphology.upload_mask_help')
            )

            # 像素大小设置
            st.write(t('morphology.pixel_size_settings'))
            pixel_size_morph = st.number_input(
                t('morphology.pixel_size_um_per_pixel'),
                min_value=0.01,
                max_value=10.0,
                value=0.65,
                step=0.01,
                key="morphology_pixel_size",
                help=t('morphology.pixel_size_input_help')
            )

            # 最小细胞面积设置
            min_area_morph = st.number_input(
                t('morphology.min_cell_area_pixels'),
                min_value=10,
                max_value=10000,
                value=100,
                step=10,
                key="morphology_min_area",
                help=t('morphology.min_area_filter_help')
            )

            # 特征提取选项
            st.write(t('morphology.feature_extraction_options'))
            use_advanced_morph = st.checkbox(
                t('morphology.use_advanced_extraction'),
                value=False,
                key="morphology_advanced",
                help=t('morphology.advanced_extraction_help')
            )

            # 高级特征选项
            if use_advanced_morph:
                with st.expander(t('morphology.advanced_feature_options'), expanded=True):
                    include_hu_morph = st.checkbox(t('morphology.hu_moments_feature'), value=True, key="morphology_hu")
                    include_intensity_morph = st.checkbox("强度特征", value=True, key="morphology_intensity")
                    include_texture_morph = st.checkbox("纹理特征", value=True, key="morphology_texture")
                    include_boundary_morph = st.checkbox("边界特征", value=True, key="morphology_boundary")
                    include_advanced_shape_morph = st.checkbox("高级形状特征", value=True, key="morphology_advanced_shape")

            # 提取按钮
            extract_button = st.button(
                t('morphology.start_feature_extraction'),
                use_container_width=True,
                type="primary",
                disabled=(uploaded_mask is None)
            )

        with col_right:
            st.write(t('morphology.results_display'))

            if extract_button and uploaded_mask is not None:
                try:
                    # 读取掩码图像
                    mask_bytes = uploaded_mask.read()
                    mask_np = np.frombuffer(mask_bytes, dtype=np.uint8)
                    mask_img = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)

                    # 如果掩码是二值图像，需要转换为标记图像
                    if len(np.unique(mask_img)) == 2:
                        # 二值图像，需要进行连通域标记
                        from skimage import measure
                        labeled_mask = measure.label(mask_img > 0)
                    else:
                        # 已经是标记图像
                        labeled_mask = mask_img

                    num_cells = len(np.unique(labeled_mask)) - 1  # 减去背景

                    if num_cells == 0:
                        st.warning(t('morphology.no_cells_in_mask'))
                    else:
                        st.info(t('morphology.cells_detected_in_mask', count=num_cells))

                        with st.spinner(t('morphology.extracting_cell_features')):
                            # 根据用户选择调用不同的特征提取函数
                            if use_advanced_morph:
                                # 使用高级特征提取（需要原始图像）
                                if uploaded_image is not None:
                                    # 读取原始图像
                                    image_bytes = uploaded_image.read()
                                    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
                                    image_decoded = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                                    # 转换为灰度图
                                    if len(image_decoded.shape) == 3:
                                        gray_image = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2GRAY)
                                    else:
                                        gray_image = image_decoded

                                    features_df = extract_advanced_cell_features(
                                        labeled_mask,
                                        image=gray_image,
                                        pixel_size=pixel_size_morph,
                                        min_area=min_area_morph,
                                        include_hu_moments=include_hu_morph,
                                        include_intensity=include_intensity_morph,
                                        include_texture=include_texture_morph,
                                        include_boundary=include_boundary_morph,
                                        include_advanced_shape=include_advanced_shape_morph
                                    )
                                    st.success(t('morphology.advanced_extraction_completed'))
                                else:
                                    st.error(t('morphology.advanced_extraction_needs_image'))
                                    features_df = None
                            else:
                                # 使用基础特征提取
                                features_df = extract_cell_features(
                                    labeled_mask,
                                    pixel_size=pixel_size_morph,
                                    min_area=min_area_morph
                                )
                                st.success(t('morphology.basic_extraction_completed'))

                            if features_df is not None and not features_df.empty:
                                # 显示特征统计
                                st.subheader(t('morphology.feature_statistics_summary'))
                                stats = get_feature_statistics(features_df)

                                # 显示关键特征的统计信息
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(t('morphology.average_area'), f"{stats['area_um2']['mean']:.1f} μm²")
                                with col2:
                                    st.metric(t('morphology.average_circularity'), f"{stats['circularity']['mean']:.3f}")
                                with col3:
                                    st.metric(t('morphology.average_major_axis'), f"{stats['major_axis_length']['mean']:.1f} μm")
                                with col4:
                                    st.metric(t('morphology.average_minor_axis'), f"{stats['minor_axis_length']['mean']:.1f} μm")

                                # 显示详细特征表格
                                with st.expander(t('morphology.view_detailed_features'), expanded=False):
                                    if use_advanced_morph:
                                        # 高级特征模式：显示所有列
                                        st.caption(t('morphology.table_scroll_hint'))
                                        st.dataframe(features_df.round(3), use_container_width=True, height=400)
                                    else:
                                        # 基础特征模式：只显示主要列
                                        display_cols = ['sequential_id', 'cell_id', 'area_um2', 'perimeter_um', 'circularity',
                                                      'major_axis_length', 'minor_axis_length', 'eccentricity',
                                                      'solidity', 'aspect_ratio']
                                        st.dataframe(features_df[display_cols].round(3), use_container_width=True, height=400)

                                # CSV下载功能
                                st.subheader(t('morphology.export_data'))

                                col_export1, col_export2 = st.columns(2)

                                with col_export1:
                                    # 下载特征CSV
                                    csv_data = features_df.to_csv(index=False)
                                    st.download_button(
                                        t('morphology.download_feature_csv'),
                                        data=csv_data,
                                        file_name=f"cell_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                        help=t('morphology.download_feature_csv_help')
                                    )

                                with col_export2:
                                    # 下载统计摘要
                                    stats_csv = pd.DataFrame(stats).T.to_csv()
                                    st.download_button(
                                        t('morphology.download_statistics_summary'),
                                        data=stats_csv,
                                        file_name=f"feature_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                        help=t('morphology.download_statistics_help')
                                    )

                            else:
                                st.warning(t('morphology.no_valid_features_extracted'))

                except Exception as e:
                    st.error(t('morphology.image_processing_error', error=str(e)))
                    st.info(t('morphology.ensure_valid_image_file'))

            else:
                if uploaded_mask is None:
                    st.info(t('messages.please_upload_mask'))
                else:
                    st.info(t('messages.click_left_extract_button'))

# ==================== 标签页8: 监督学习 ====================
with tab8:
    st.header(t('ml.supervised_learning_title'))
    st.markdown(t('ml.supervised_learning_description'))

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader(t('ml.settings_subheader'))

        # 数据上传
        uploaded_data = st.file_uploader(
            t('ml.upload_feature_data_csv'),
            type=["csv"],
            key="supervised_data_upload",
            help=t('ml.upload_feature_data_help')
        )

        if uploaded_data is not None:
            try:
                data_df = pd.read_csv(uploaded_data)
                st.success(t('ml.data_loaded_success', rows=data_df.shape[0], cols=data_df.shape[1]))

                # 显示数据预览
                with st.expander(t('ml.data_preview'), expanded=False):
                    st.dataframe(data_df.head(10), use_container_width=True)

                # 目标列选择
                st.write(t('ml.target_variable_settings'))
                target_column = st.selectbox(
                    t('ml.select_target_column'),
                    options=data_df.columns.tolist(),
                    index=len(data_df.columns) - 1,
                    help=t('ml.select_target_column_help')
                )

                # 任务类型
                task_type = st.radio(
                    t('ml.task_type'),
                    options=["auto", "classification", "regression"],
                    index=0,
                    help=t('ml.task_type_help')
                )

                # 模型选择
                st.write(t('ml.model_settings'))
                use_automl = st.checkbox(t('ml.use_automl'), value=True)

                if not use_automl:
                    model_options = ['random_forest', 'gradient_boosting', 'svm', 'logistic', 'xgboost']
                    model_name = st.selectbox(t('ml.select_model'), options=model_options)
                else:
                    model_name = None

                # 高级设置
                with st.expander(t('ml.advanced_settings'), expanded=False):
                    test_size = st.slider(t('ml.test_size'), 0.1, 0.5, 0.2, 0.05)
                    cv_folds = st.slider(t('ml.cv_folds'), 3, 10, 5, 1)

                    feature_selection = st.selectbox(
                        t('ml.feature_selection_method'),
                        options=[None, "correlation", "mutual_info", "rfe", "tree_based"],
                        index=0
                    )

                    feature_scaling = st.selectbox(
                        t('ml.feature_scaling_method'),
                        options=["standard", "minmax", "robust", None],
                        index=0
                    )

                    hyperparameter_tuning = st.checkbox(t('ml.hyperparameter_tuning'), value=False)

                # 训练按钮
                if st.button(t('ml.start_training'), type="primary", use_container_width=True):
                    with st.spinner(t('ml.training_model')):
                        try:
                            if use_automl:
                                # AutoML模式
                                best_model, comparison_df, results = compare_models_automl(
                                    data_df=data_df,
                                    target_column=target_column,
                                    task_type=task_type,
                                    test_size=test_size,
                                    feature_selection=feature_selection,
                                    feature_scaling=feature_scaling,
                                    cv_folds=cv_folds,
                                    tune_best=hyperparameter_tuning
                                )

                                st.session_state['supervised_model'] = best_model
                                st.session_state['supervised_results'] = results
                                st.session_state['model_comparison'] = comparison_df

                            else:
                                # 单模型训练
                                model, results = train_supervised_model(
                                    data_df=data_df,
                                    target_column=target_column,
                                    task_type=task_type,
                                    test_size=test_size,
                                    model_name=model_name,
                                    feature_selection=feature_selection,
                                    feature_scaling=feature_scaling,
                                    hyperparameter_tuning=hyperparameter_tuning,
                                    cv_folds=cv_folds
                                )

                                st.session_state['supervised_model'] = model
                                st.session_state['supervised_results'] = results

                            st.success(t('ml.model_training_completed'))

                        except Exception as e:
                            st.error(t('ml.training_failed', error=str(e)))

            except Exception as e:
                st.error(t('ml.data_loading_failed', error=str(e)))

    with col_right:
        st.subheader(t('ml.training_results'))

        if 'supervised_results' in st.session_state:
            results = st.session_state['supervised_results']

            # 显示模型信息
            st.write(t('ml.model_info'))
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(t('ml.task_type_metric'), results['task_type'])
            with col2:
                st.metric(t('ml.model_metric'), results.get('best_model_name', results.get('model_name', 'N/A')))
            with col3:
                st.metric(t('ml.n_features_metric'), results['n_features'])

            # 模型对比（AutoML模式）
            if 'model_comparison' in st.session_state:
                st.write(t('ml.model_comparison'))
                st.dataframe(st.session_state['model_comparison'], use_container_width=True)

            # 评估指标
            st.write(t('ml.evaluation_metrics'))
            metrics = results['metrics']

            if results['task_type'] == 'classification':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(t('metrics.accuracy'), f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric(t('metrics.precision'), f"{metrics['precision']:.4f}")
                with col3:
                    st.metric(t('metrics.recall'), f"{metrics['recall']:.4f}")
                with col4:
                    st.metric(t('metrics.f1_score'), f"{metrics['f1_score']:.4f}")

                # 混淆矩阵
                st.write(t('ml.confusion_matrix'))
                fig_cm = plot_confusion_matrix(
                    results['y_test'],
                    results['predictions']
                )
                st.pyplot(fig_cm)

            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(t('metrics.r2_score'), f"{metrics['r2_score']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                with col3:
                    st.metric("MAE", f"{metrics['mae']:.4f}")
                with col4:
                    if metrics.get('mape'):
                        st.metric("MAPE", f"{metrics['mape']:.4f}")

                # 预测vs实际
                st.write(t('ml.prediction_vs_actual'))
                fig_pred = plot_prediction_vs_actual(
                    results['y_test'],
                    results['predictions']
                )
                st.pyplot(fig_pred)

            # 特征重要性
            if hasattr(st.session_state['supervised_model'], 'feature_importances_'):
                st.write(t('ml.feature_importance'))
                fig_imp = plot_feature_importance(
                    st.session_state['supervised_model'],
                    results['feature_names'],
                    top_n=15
                )
                st.pyplot(fig_imp)

            # 模型保存
            st.write(t('ml.save_model_section'))
            model_name_save = st.text_input(t('ml.model_name_input'), value="cell_model")
            if st.button(t('ml.save_model_button'), use_container_width=True):
                save_path = f"models/{model_name_save}.pkl"
                success = save_model(
                    st.session_state['supervised_model'],
                    save_path,
                    metadata=results['metrics'],
                    feature_names=results['feature_names'],
                    scaler=results.get('scaler'),
                    task_type=results['task_type']
                )
                if success:
                    st.success(t('ml.model_saved_success', path=save_path))
                else:
                    st.error(t('ml.model_save_failed'))

        else:
            st.info(t('ml.please_upload_and_train'))

# ==================== 标签页9: 主动学习 ====================
with tab9:
    st.header(t('ml.active_learning_title'))
    st.markdown(t('ml.active_learning_description'))

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader(t('common.settings'))

        # 数据上传
        st.write(t('ml.data_upload_section'))
        uploaded_train = st.file_uploader(
            t('ml.upload_train_data'),
            type=["csv"],
            key="active_train_upload",
            help=t('ml.upload_train_data_help')
        )

        uploaded_pool = st.file_uploader(
            t('ml.upload_pool_data'),
            type=["csv"],
            key="active_pool_upload",
            help=t('ml.upload_pool_data_help')
        )

        if uploaded_train is not None and uploaded_pool is not None:
            try:
                train_df = pd.read_csv(uploaded_train)
                pool_df = pd.read_csv(uploaded_pool)

                st.success(t('ml.train_set_info', count=train_df.shape[0]))
                st.success(t('ml.pool_set_info', count=pool_df.shape[0]))

                # 目标列选择
                st.write("**🎯 目标变量设置**")
                target_column = st.selectbox(
                    "选择目标列",
                    options=train_df.columns.tolist(),
                    index=len(train_df.columns) - 1,
                    key="active_target"
                )

                # 任务类型
                task_type = st.radio(
                    t('ml.task_type'),
                    options=["classification", "regression"],
                    index=0,
                    key="active_task"
                )

                # 策略选择
                st.write(t('ml.active_learning_strategy_section'))
                strategy = st.selectbox(
                    t('ml.sampling_strategy'),
                    options=["uncertainty", "qbc", "random"],
                    format_func=lambda x: {
                        "uncertainty": t('ml.uncertainty_sampling'),
                        "qbc": t('ml.qbc_sampling'),
                        "random": t('ml.random_sampling')
                    }[x]
                )

                # 模型选择
                model_name = st.selectbox(
                    t('ml.base_model'),
                    options=["random_forest", "gradient_boosting", "svm", "logistic"],
                    index=0,
                    key="active_model"
                )

                # 迭代设置
                st.write(t('ml.iteration_settings_section'))
                n_iterations = st.slider(t('ml.n_iterations'), 5, 50, 10, 5)
                samples_per_iteration = st.slider(t('ml.samples_per_iteration'), 5, 50, 10, 5)

                # 开始主动学习
                if st.button(t('ml.start_active_learning'), type="primary", use_container_width=True):
                    with st.spinner(t('ml.executing_active_learning')):
                        try:
                            # 分离特征和标签
                            X_train = train_df.drop(columns=[target_column]).values
                            y_train = train_df[target_column].values
                            X_pool = pool_df.drop(columns=[target_column]).values
                            y_pool = pool_df[target_column].values

                            # 执行主动学习
                            results = active_learning_workflow(
                                X_train_initial=X_train,
                                y_train_initial=y_train,
                                X_pool=X_pool,
                                y_pool_true=y_pool,
                                model_name=model_name,
                                task_type=task_type,
                                strategy=strategy,
                                n_iterations=n_iterations,
                                samples_per_iteration=samples_per_iteration
                            )

                            st.session_state['active_results'] = results
                            st.success(t('messages.active_learning_completed'))

                        except Exception as e:
                            st.error(t('ml.active_learning_failed', error=str(e)))

            except Exception as e:
                st.error(t('ml.data_loading_failed', error=str(e)))

    with col_right:
        st.subheader(t('ml.learning_progress'))

        if 'active_results' in st.session_state:
            results = st.session_state['active_results']

            # 显示基本信息
            st.write(t('ml.learning_statistics'))
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(t('ml.n_iterations'), results['n_iterations'])
            with col2:
                st.metric(t('metrics.strategy'), results['strategy'])
            with col3:
                st.metric(t('metrics.task_type'), results['task_type'])

            # 学习曲线
            st.write(t('ml.performance_curve'))
            fig_traj = plot_optimization_trajectory(results, metric='test_score')
            st.pyplot(fig_traj)

            # 收敛图
            st.write(t('ml.convergence_analysis'))
            fig_conv = plot_convergence(results, show_confidence=True)
            st.pyplot(fig_conv)

            # 迭代详情
            with st.expander(t('ml.iteration_details'), expanded=False):
                metrics_df = pd.DataFrame(results['iteration_metrics'])
                st.dataframe(metrics_df, use_container_width=True)

            # 最终模型性能
            st.write(t('ml.final_model_performance'))
            final_metrics = results['iteration_metrics'][-1]
            col1, col2 = st.columns(2)
            with col1:
                st.metric(t('metrics.train_score'), f"{final_metrics['train_score']:.4f}")
            with col2:
                st.metric(t('metrics.test_score'), f"{final_metrics['test_score']:.4f}")

        else:
            st.info(t('ml.please_upload_and_start'))

# ==================== 标签页10: 虚拟筛选 ====================
with tab10:
    st.header(t('ml.virtual_screening_title'))
    st.markdown(t('ml.virtual_screening_description'))

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader(t('common.settings'))

        # 模型加载
        st.write(t('ml.model_loading_section'))
        model_source = st.radio(
            t('ml.model_source'),
            options=["use_current", "upload_saved"],
            format_func=lambda x: t('ml.use_current_model') if x == "use_current" else t('ml.upload_saved_model'),
            index=0
        )

        model_loaded = False
        model_path = None

        if model_source == "use_current":
            if 'supervised_model' in st.session_state:
                st.success(t('messages.model_loaded'))
                model_loaded = True
            else:
                st.warning(t('messages.please_train_model_first'))
        else:
            uploaded_model = st.file_uploader(
                t('ml.upload_model_file'),
                type=["pkl", "joblib"],
                key="screening_model_upload"
            )
            if uploaded_model is not None:
                # 保存临时文件
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(uploaded_model.read())
                    model_path = tmp_file.name
                st.success(t('messages.model_file_uploaded'))
                model_loaded = True

        # 数据上传
        st.write(t('ml.data_upload_section'))
        uploaded_screen_data = st.file_uploader(
            t('ml.upload_screening_data'),
            type=["csv"],
            key="screening_data_upload",
            help=t('ml.upload_screening_data_help')
        )

        if model_loaded and uploaded_screen_data is not None:
            try:
                screen_df = pd.read_csv(uploaded_screen_data)
                st.success(t('ml.screening_data_loaded', count=screen_df.shape[0]))

                # 筛选设置
                st.write(t('ml.screening_settings_section'))
                min_confidence = st.slider(
                    t('ml.min_confidence_threshold'),
                    0.0, 1.0, 0.7, 0.05,
                    help=t('ml.min_confidence_threshold_help')
                )

                top_n = st.number_input(
                    t('ml.select_top_n_candidates'),
                    min_value=10,
                    max_value=1000,
                    value=100,
                    step=10
                )

                ranking_criteria = st.selectbox(
                    t('ml.ranking_criteria'),
                    options=["prediction", "confidence", "combined"],
                    format_func=lambda x: {
                        "prediction": t('ml.prediction_value'),
                        "confidence": t('ml.confidence_value'),
                        "combined": t('ml.combined_score')
                    }[x]
                )

                # 开始筛选
                if st.button(t('ml.start_screening'), type="primary", use_container_width=True):
                    with st.spinner(t('common.performing_screening')):
                        try:
                            if model_source == "use_current":
                                # 使用session中的模型
                                model = st.session_state['supervised_model']
                                results_info = st.session_state['supervised_results']

                                # 手动预测
                                feature_names = results_info['feature_names']
                                X_screen = screen_df[feature_names].values

                                # 应用缩放器
                                if results_info.get('scaler'):
                                    X_screen = results_info['scaler'].transform(X_screen)

                                # 预测
                                predictions = model.predict(X_screen)
                                screen_df['prediction'] = predictions

                                # 计算置信度
                                if hasattr(model, 'predict_proba'):
                                    probabilities = model.predict_proba(X_screen)
                                    confidence = np.max(probabilities, axis=1)
                                else:
                                    confidence = np.ones(len(predictions))

                                screen_df['confidence'] = confidence
                                results_df = screen_df

                            else:
                                # 使用上传的模型
                                results_df, info = screen_dataset(
                                    model_path=model_path,
                                    data_df=screen_df,
                                    min_confidence=None,
                                    return_probabilities=True
                                )

                            # 过滤和排序
                            results_df = filter_by_confidence(
                                results_df,
                                min_confidence=min_confidence
                            )

                            results_df = rank_by_prediction(
                                results_df,
                                ascending=False
                            )

                            # 选择Top N
                            top_candidates = select_top_candidates(
                                results_df,
                                n_candidates=top_n,
                                criteria=ranking_criteria,
                                confidence_threshold=min_confidence
                            )

                            st.session_state['screening_results'] = results_df
                            st.session_state['top_candidates'] = top_candidates
                            st.success(t('ml.screening_completed', count=len(results_df)))

                        except Exception as e:
                            st.error(t('ml.screening_failed', error=str(e)))
                            import traceback
                            st.error(traceback.format_exc())

            except Exception as e:
                st.error(t('ml.data_loading_failed', error=str(e)))

    with col_right:
        st.subheader(t('ml.screening_results'))

        if 'screening_results' in st.session_state:
            results_df = st.session_state['screening_results']
            top_candidates = st.session_state.get('top_candidates', results_df.head(100))

            # 统计信息
            st.write(t('ml.screening_statistics'))
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(t('ml.total_samples'), len(results_df))
            with col2:
                st.metric(t('ml.average_confidence'), f"{results_df['confidence'].mean():.3f}")
            with col3:
                st.metric(t('ml.top_candidates'), len(top_candidates))

            # 预测分布
            st.write(t('ml.prediction_distribution'))
            fig_pred_dist = plot_prediction_distribution(
                results_df,
                task_type='regression'
            )
            st.pyplot(fig_pred_dist)

            # 置信度分布
            st.write(t('ml.confidence_distribution'))
            fig_conf_dist = plot_confidence_distribution(results_df)
            st.pyplot(fig_conf_dist)

            # Top候选物可视化
            st.write(t('ml.top_candidates_visualization'))
            fig_top = plot_top_candidates(
                top_candidates,
                top_n=min(20, len(top_candidates))
            )
            st.pyplot(fig_top)

            # 预测vs置信度
            st.write(t('ml.prediction_vs_confidence'))
            fig_pred_conf = plot_prediction_vs_confidence(results_df)
            st.pyplot(fig_pred_conf)

            # 结果表格
            with st.expander(t('ml.view_detailed_results'), expanded=False):
                st.dataframe(
                    top_candidates.round(4),
                    use_container_width=True,
                    height=400
                )

            # 导出结果
            st.write(t('common.export_results'))
            col1, col2 = st.columns(2)

            with col1:
                # 导出所有结果
                csv_all = results_df.to_csv(index=False)
                st.download_button(
                    t('ml.download_all_results'),
                    data=csv_all,
                    file_name=f"screening_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                # 导出Top候选物
                csv_top = top_candidates.to_csv(index=False)
                st.download_button(
                    t('ml.download_top_candidates'),
                    data=csv_top,
                    file_name=f"screening_top_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        else:
            st.info(t('ml.please_load_model_and_upload'))
