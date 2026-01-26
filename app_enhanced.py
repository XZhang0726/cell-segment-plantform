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
                GPU_WARNING = f"⚠️ {GPU_NAME} 需要PyTorch 2.10.0+cu128支持，当前版本 {pytorch_version} 不兼容。"
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
    page_title="细胞分割平台 - 增强版",
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
        "Otsu阈值": SegmentationMethod.OTSU,
        "自适应阈值": SegmentationMethod.ADAPTIVE,
        "分水岭算法": SegmentationMethod.WATERSHED,
        "Canny边缘检测": SegmentationMethod.EDGE_CANNY,
        "Cellpose深度学习": SegmentationMethod.CELLPOSE,
        "CellViT深度学习": SegmentationMethod.CELLVIT,
        "CellSAM深度学习": SegmentationMethod.CELLSAM
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
    st.title("🔬 细胞分割平台 - 增强版")
    st.markdown("支持批量处理、多方法对比、预处理选项和结果导出")

    # GPU状态指示器
    if GPU_AVAILABLE and GPU_COMPATIBLE:
        st.success(f"🚀 GPU加速可用: {GPU_NAME}")
    elif GPU_AVAILABLE and not GPU_COMPATIBLE:
        st.error(f"❌ GPU检测到但不兼容: {GPU_NAME}")
        st.warning(GPU_WARNING)
    else:
        st.info("ℹ️ GPU不可用，将使用CPU处理")
with col_help:
    st.write("")  # 添加空行对齐
    with st.popover("📖 使用说明"):
        st.markdown("""
        ### 功能介绍

        #### 1. 图像分割
        - **分割方法**:
          - 传统方法: Otsu阈值、自适应阈值、分水岭算法、Canny边缘检测
          - 深度学习: Cellpose（推荐用于复杂细胞图像和重叠细胞分割）
        - **预处理选项**: 去噪、对比度增强、归一化
        - **结果导出**: 下载分割掩码和叠加图

        #### 2. 对比模式
        - 同时使用多种分割方法进行对比
        - 并排显示不同方法的分割结果
        - 性能对比表：显示前景比例和处理时间
        - 批量导出所有方法的掩码、叠加图和对比报告

        #### 3. 批量处理
        - 一次上传多张图像进行批量分割
        - 自动应用相同的分割参数和预处理选项
        - 实时显示处理进度
        - 可视化查看每张图像的分割结果
        - 批量导出掩码、叠加图和统计报告（ZIP格式）

        #### 4. 预处理选项
        - **去噪处理**: 使用高斯滤波减少图像噪声
        - **对比度增强**: 使用CLAHE算法增强局部对比度
        - **归一化**: 将像素值归一化到标准范围

        #### 5. 后处理选项
        - **区域闭合**: 使用形态学闭运算填充细胞边界间隙，获得完整的细胞区域
        - **提取单个细胞**: 自动提取每个细胞样本，支持机器学习训练数据准备
        - **最小细胞面积**: 过滤掉面积过小的噪声区域

        #### 6. 异常检测
        - **独立模块**: 在"🔍 异常检测"tab中上传细胞特征CSV文件
        - **异常检测算法**: Isolation Forest、LOF、One-Class SVM、Elliptic Envelope，识别形态异常的细胞
        - **降维可视化**: PCA、t-SNE、UMAP，2D可视化异常样本分布
        - **结果导出**: 下载带异常标签的CSV文件，支持导出仅正常样本

        #### 7. 聚类分析
        - **独立模块**: 在"📊 聚类分析"tab中上传细胞特征CSV文件
        - **聚类算法**: K-means、DBSCAN、层次聚类、GMM，自动发现细胞亚群
        - **左右分栏布局**: 左侧设置参数和查看结果，右侧实时可视化聚类效果
        - **降维可视化**: PCA、t-SNE、UMAP，2D可视化高维特征
        - **结果导出**: 下载带聚类标签的CSV文件
        - **CSV格式**: 需包含基础形态学特征（面积、圆度、长轴、短轴等）

        ### 使用技巧
        - **Cellpose深度学习**: 对于重叠或接触的细胞，推荐使用Cellpose方法，效果最佳
        - **模型选择**: cyto2适合大多数细胞质染色图像，nuclei适合细胞核染色图像
        - **预处理**: 对于噪声较大的图像，建议启用去噪和对比度增强
        - **区域闭合**: 默认启用，可填补细胞边界的小间隙，获得更完整的分割结果
        - **细胞提取**: 启用后可导出单个细胞样本，适合用于深度学习模型训练
        - **批量处理**: 适合处理大量相似类型的细胞图像
        - **对比模式**: 不确定哪种方法最适合时，使用"对比模式"tab快速评估多种方法
        - **参数调整**: 根据图像特点调整方法参数以获得最佳效果
        - **异常检测**: 完成特征提取后，可在"🔍 异常检测"tab上传CSV进行异常检测，识别形态异常的细胞
        - **聚类分析**: 完成特征提取后，可在"📊 聚类分析"tab上传CSV进行聚类分析，自动发现细胞亚群
        """)

# ==================== 模型融合流程函数 ====================
def run_fusion_pipeline(image, selected_models, strategy, iou_threshold, min_vote_count, weights, model_params, display_col,
                       model_reliabilities=None, conflict_threshold=0.6):
    """执行完整的融合流程（支持简单策略和DST高级融合）"""
    # 使用优化版本的实例匹配（57倍加速）
    from src.core.fusion import match_instances, fuse_instances, fuse_instances_dst, generate_confidence_maps
    from src.core.fusion.uncertainty import compute_disagreement_map, compute_model_consistency
    from skimage.color import label2rgb
    import matplotlib.pyplot as plt

    # 模型名称映射
    model_name_mapping = {
        "cellpose": "Cellpose深度学习",
        "cellvit": "CellViT深度学习",
        "cellsam": "CellSAM深度学习",
        "watershed": "分水岭算法",
        "otsu": "Otsu阈值",
        "adaptive": "自适应阈值",
        "canny": "Canny边缘检测"
    }

    with display_col:
        with st.spinner("正在运行多模型推理..."):
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

            postprocess_options = {
                'closing': True,
                'closing_kernel_size': 5,
                'extract_cells': False,
                'min_cell_area': 100,
                'extract_morphology': False
            }

            for idx, model_name in enumerate(selected_models):
                st.text(f"运行 {model_name}...")

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
                    st.error(f"❌ {model_name} 推理失败: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())
                    continue

                progress_bar.progress((idx + 1) / len(selected_models))

            if len(masks_list) < 2:
                st.error("❌ 至少需要2个模型成功运行才能进行融合")
                return

            st.success(f"✅ 完成 {len(masks_list)} 个模型的推理")

        # 创建融合进度条
        st.subheader("🔀 融合处理进度")
        fusion_progress = st.progress(0)
        fusion_status = st.empty()

        # 步骤1: 实例匹配 (0% -> 33%)
        fusion_status.text("🔍 步骤 1/3: 正在进行实例匹配...")
        try:
            matched_groups = match_instances(masks_list, iou_threshold)
            fusion_progress.progress(0.33)
            fusion_status.text(f"✅ 步骤 1/3 完成: 匹配到 {len(matched_groups)} 个细胞实例组")
        except Exception as e:
            st.error(f"❌ 实例匹配失败: {str(e)}")
            return

        # 步骤2: 融合掩码 (33% -> 66%)
        fusion_status.text("🎯 步骤 2/3: 正在融合分割结果...")
        try:
            # 判断使用简单策略还是DST高级融合
            if strategy == 'dempster_shafer':
                # DST高级融合
                fusion_status.text("🎓 使用Dempster-Shafer理论进行高级融合...")

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

                fusion_status.text(f"✅ 步骤 2/3 完成: DST融合生成 {np.max(fused_mask)} 个细胞 "
                                 f"(平均冲突={dst_stats['average_conflict']:.3f}, 主要策略={dominant_strategy})")
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
                fusion_status.text(f"✅ 步骤 2/3 完成: 融合生成 {np.max(fused_mask)} 个细胞")
        except Exception as e:
            st.error(f"❌ 融合失败: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            return

        # 步骤3: 计算不确定性 (66% -> 100%)
        fusion_status.text("📊 步骤 3/3: 正在计算模型一致性和不确定性...")
        try:
            disagreement_map, consistency_score = compute_disagreement_map(masks_list)
            consistency_matrix, avg_consistency = compute_model_consistency(masks_list)
            fusion_progress.progress(1.0)
            fusion_status.text(f"✅ 步骤 3/3 完成: 模型一致性 {consistency_score:.2%}")
        except Exception as e:
            st.error(f"❌ 不确定性计算失败: {str(e)}")
            return

        # 完成提示
        st.success("🎉 融合处理全部完成！")

        # 清除进度条和状态文本（可选，如果想保留就注释掉）
        # fusion_progress.empty()
        # fusion_status.empty()

        # 5. 显示结果
        st.subheader("📊 融合结果")

        # 根据是否使用DST决定显示哪些tab
        if dst_stats is not None:
            result_tabs = st.tabs(["融合掩码", "不确定性热图", "模型对比", "🎓 DST分析"])
        else:
            result_tabs = st.tabs(["融合掩码", "不确定性热图", "模型对比"])

        with result_tabs[0]:
            # 显示融合掩码
            try:
                fused_display = label2rgb(fused_mask, bg_label=0)
                st.image(fused_display, caption="融合后的分割结果", use_container_width=True)

                # 添加下载按钮
                from io import BytesIO
                fused_img = Image.fromarray((fused_display * 255).astype(np.uint8))
                buf = BytesIO()
                fused_img.save(buf, format='PNG')
                st.download_button(
                    label="📥 下载融合掩码",
                    data=buf.getvalue(),
                    file_name="fused_mask.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"显示融合掩码失败: {str(e)}")

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
                    label="📥 下载不确定性热图",
                    data=buf.getvalue(),
                    file_name="uncertainty_heatmap.png",
                    mime="image/png"
                )

                plt.close(fig)
            except Exception as e:
                st.error(f"显示不确定性热图失败: {str(e)}")

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
                            label=f"📥 下载",
                            data=buf.getvalue(),
                            file_name=f"{model_name}_result.png",
                            mime="image/png",
                            key=f"download_{model_name}_{idx}"
                        )
            except Exception as e:
                st.error(f"显示模型对比失败: {str(e)}")

        # DST分析tab（仅在使用DST融合时显示）
        if dst_stats is not None:
            with result_tabs[3]:
                st.markdown("### 🎓 Dempster-Shafer理论融合分析")

                # DST统计摘要
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("融合实例数", dst_stats['fused_count'])
                with col2:
                    st.metric("平均冲突度", f"{dst_stats['average_conflict']:.3f}")
                with col3:
                    st.metric("平均不确定性", f"{dst_stats['average_uncertainty']:.3f}")
                with col4:
                    st.metric("高冲突实例", dst_stats['high_conflict_count'])

                # 策略分布统计
                st.markdown("#### 📊 自适应融合策略分布")
                st.caption("DST根据置信度和冲突度自动选择不同的融合策略")

                # 显示置信度和冲突度分布
                if 'confidence_distribution' in dst_stats and 'conflict_distribution' in dst_stats:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**置信度分布**")
                        conf_dist = dst_stats['confidence_distribution']
                        st.text(f"范围: [{conf_dist['min']:.3f}, {conf_dist['max']:.3f}]")
                        st.text(f"均值: {conf_dist['mean']:.3f}")
                        st.text(f"标准差: {conf_dist['std']:.3f}")

                        # 判断置信度变化是否足够
                        if conf_dist['std'] < 0.05:
                            st.warning("⚠️ 置信度变化很小，可能导致策略单一")

                    with col2:
                        st.markdown("**冲突度分布**")
                        conflict_dist = dst_stats['conflict_distribution']
                        st.text(f"范围: [{conflict_dist['min']:.3f}, {conflict_dist['max']:.3f}]")
                        st.text(f"均值: {conflict_dist['mean']:.3f}")
                        st.text(f"标准差: {conflict_dist['std']:.3f}")

                        # 判断冲突度变化是否足够
                        if conflict_dist['std'] < 0.05:
                            st.warning("⚠️ 冲突度变化很小，可能导致策略单一")

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
                    with st.expander("📖 策略说明", expanded=False):
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
                    st.warning("⚠️ 未找到策略分布数据")

                # 高冲突实例列表
                if dst_stats['high_conflict_count'] > 0:
                    st.markdown("#### ⚠️ 高冲突实例（需要注意）")
                    st.caption("这些实例的模型预测存在较大分歧，建议人工审查")

                    import pandas as pd
                    conflict_df = pd.DataFrame(dst_stats['high_conflict_instances'])
                    st.dataframe(conflict_df, use_container_width=True)

                # 分水岭边界细化统计
                watershed_stats = dst_stats.get('watershed_refinement', {})
                if watershed_stats.get('refined', False):
                    st.markdown("#### 🌊 分水岭边界细化")
                    st.success(f"✅ 成功细化 {watershed_stats['refined_pixels']} 个像素 "
                              f"({watershed_stats['refined_percentage']:.2f}%)")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("种子数量", watershed_stats.get('num_markers', 0))
                    with col2:
                        st.metric("高冲突像素", watershed_stats.get('high_conflict_pixels', 0))
                elif watershed_stats:
                    reason = watershed_stats.get('reason', 'unknown')
                    if reason == 'no_image':
                        st.info("ℹ️ 分水岭细化：未提供原始图像")
                    elif reason == 'no_markers':
                        st.warning("⚠️ 分水岭细化：未找到确定种子")
                    elif reason == 'disabled_or_no_conflict':
                        st.info("ℹ️ 分水岭细化：未启用或无高冲突区域")
                    else:
                        st.warning(f"⚠️ 分水岭细化失败：{reason}")

                # 详细融合结果
                with st.expander("📋 查看详细融合结果", expanded=False):
                    if len(dst_stats['fusion_results']) > 0:
                        import pandas as pd
                        results_df = pd.DataFrame(dst_stats['fusion_results'])
                        st.dataframe(results_df, use_container_width=True)

                        # 下载按钮
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 下载DST融合结果CSV",
                            data=csv,
                            file_name="dst_fusion_results.csv",
                            mime="text/csv"
                        )

        # 6. 统计信息
        st.subheader("📈 统计信息")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("检测细胞数", np.max(fused_mask))
        with col2:
            st.metric("模型一致性", f"{consistency_score:.2%}")
        with col3:
            st.metric("平均模型IoU", f"{avg_consistency:.2%}")

        # 7. 导出选项
        st.subheader("💾 导出结果")
        st.info("导出功能开发中...")


# 创建标签页
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📤 图像分割", "🔍 对比模式", "🔀 模型融合", "📦 批量处理", "🔬 细胞形态学提取", "🔍 异常检测", "📊 聚类分析"])

# ==================== 标签页1: 图像分割 ====================
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.header("⚙️ 设置")

        # 图像上传
        uploaded_file = st.file_uploader(
            "上传细胞图像",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="single_upload"
        )

        # 像素大小设置
        st.write("**📏 像素大小设置**")
        pixel_size = st.number_input(
            "像素大小 (μm/pixel)",
            min_value=0.01,
            max_value=10.0,
            value=0.65,
            step=0.01,
            help="输入显微镜的像素大小，用于计算实际物理尺寸"
        )
        st.session_state['pixel_size'] = pixel_size

        # 预处理选项
        with st.expander("🔧 预处理选项", expanded=False):
            denoise = st.checkbox("去噪处理", value=False, help="使用高斯滤波去除噪声")
            enhance = st.checkbox("对比度增强", value=False, help="使用CLAHE增强对比度")
            normalize = st.checkbox("归一化", value=False, help="归一化像素值到0-255")

        # 后处理选项
        with st.expander("🔬 后处理选项", expanded=False):
            closing = st.checkbox("区域闭合", value=True, help="使用形态学闭运算填充细胞边界间隙")
            if closing:
                closing_kernel_size = st.slider("闭运算核大小", 3, 15, 3, 2, help="核越大，填充的间隙越大")
            else:
                closing_kernel_size = 5

            extract_cells = st.checkbox("提取单个细胞", value=False, help="提取并保存单个细胞样本，用于机器学习训练")
            if extract_cells:
                min_cell_area = st.slider("最小细胞面积", 50, 500, 50, 10, help="过滤掉面积小于此值的区域")

            extract_morphology = st.checkbox("提取形态学特征", value=False, help="提取单个细胞的几何形态学特征（面积、周长、圆度等）")

            # 高级特征提取选项
            use_advanced_features = st.checkbox("使用高级特征提取", value=False, help="提取更高级的形态学、纹理和强度特征")
            if use_advanced_features:
                st.caption("**高级特征类别**")
                include_hu_moments = st.checkbox("Hu矩特征", value=True, help="7个旋转、缩放、平移不变的形状描述符")
                include_intensity = st.checkbox("强度统计特征", value=True, help="灰度统计（均值、标准差、偏度、峰度、熵）")
                include_texture = st.checkbox("纹理特征(GLCM)", value=True, help="基于灰度共生矩阵的Haralick纹理特征")
                include_boundary = st.checkbox("边界复杂度特征", value=True, help="边界粗糙度、凹凸性分析")
                include_advanced_shape = st.checkbox("高级形状特征", value=True, help="椭圆度、伸长度、分形维数等")

        # 分割方法选择
        st.subheader("📐 分割方法")
        method = st.selectbox(
            "选择方法",
            ["Otsu阈值", "自适应阈值", "分水岭算法", "Canny边缘检测", "Cellpose深度学习", "CellViT深度学习", "CellSAM深度学习"],
            index=4  # 默认选择Cellpose深度学习
        )

        # 方法参数（直接显示，不使用折叠面板）
        if method == "自适应阈值":
            st.write("**方法参数**")
            block_size = st.slider("块大小", 3, 51, 11, 2)
            C = st.slider("常数C", 0, 20, 2)
            params = {"block_size": block_size, "C": C}
        elif method == "Canny边缘检测":
            st.write("**方法参数**")
            low_threshold = st.slider("低阈值", 0, 200, 50, 10)
            high_threshold = st.slider("高阈值", 0, 300, 150, 10)
            params = {"low_threshold": low_threshold, "high_threshold": high_threshold}
        elif method == "Cellpose深度学习":
            st.write("**方法参数**")
            model_type = st.selectbox("模型类型", ["cyto2", "cyto", "nuclei"],
                                     help="cyto2: 细胞质模型(推荐), cyto: 旧版细胞质模型, nuclei: 细胞核模型")
            diameter = st.slider("细胞直径(像素)", 0, 100, 30, 5,
                                help="设置为0则自动检测")
            if diameter == 0:
                diameter = None

            # GPU选项
            if GPU_AVAILABLE and GPU_COMPATIBLE:
                use_gpu = st.checkbox(f"🚀 使用GPU加速 ({GPU_NAME})", value=True,
                                     help="启用GPU可大幅提升处理速度（10-50倍）")
            elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                use_gpu = False
                st.error(f"⚠️ GPU不兼容: {GPU_WARNING}")
                st.info("💡 将使用CPU模式处理（速度较慢但稳定）")
            else:
                use_gpu = False
                st.info("ℹ️ GPU不可用，将使用CPU处理")

            # 高级参数
            with st.expander("⚙️ 高级参数", expanded=False):
                batch_size = st.slider("批处理大小", 1, 64, 8, 1,
                                      help="处理大图像时的批处理大小，增大可提高速度但需要更多内存")
                use_normalize = st.checkbox("启用图像归一化", value=True,
                                           help="对图像进行归一化处理，可改善分割质量")
                if use_normalize:
                    tile_norm_blocksize = st.slider("归一化块大小", 0, 256, 64, 16,
                                                   help="0表示全局归一化，>0表示分块归一化")
                    normalize = {"tile_norm_blocksize": tile_norm_blocksize}
                else:
                    normalize = None

            params = {"model_type": model_type, "diameter": diameter, "use_gpu": use_gpu,
                     "batch_size": batch_size, "normalize": normalize}
        elif method == "CellViT深度学习":
            st.write("**方法参数**")

            # 环境检查
            if not CELLVIT_ENV_OK:
                st.error(f"⚠️ CellViT专用环境未找到！")
                st.warning("CellViT需要专用环境。请按以下步骤创建：")
                st.code("conda create --prefix ./env_cellvit python=3.12 -y\nsource activate ./env_cellvit\npip install cellvit torch torchvision", language="bash")
                st.info("💡 或者选择其他分割方法（如Cellpose）")
            else:
                st.success(f"✅ CellViT专用环境已就绪 (将自动调用 {CELLVIT_ENV_STATUS})")

            model_type = st.selectbox("模型类型", ["CellViT-256"],
                                     help="CellViT-256: 基于Vision Transformer的细胞核分割模型")
            target_size = st.slider("目标图像大小", 256, 1024, 512, 64,
                                   help="图像会被调整到此大小进行处理。较大的值可检测更多小细胞，但处理更慢。推荐512-768")

            # GPU选项
            if GPU_AVAILABLE and GPU_COMPATIBLE:
                use_gpu = st.checkbox(f"🚀 使用GPU加速 ({GPU_NAME})", value=True,
                                     help="CellViT推荐使用GPU（需要8GB+ VRAM）")
            elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                use_gpu = False
                st.error(f"⚠️ GPU不兼容: {GPU_WARNING}")
                st.info("💡 将使用CPU模式处理（速度较慢）")
            else:
                use_gpu = False
                st.info("ℹ️ GPU不可用，将使用CPU处理")

            params = {"model_type": model_type, "target_size": target_size, "use_gpu": use_gpu}
        elif method == "CellSAM深度学习":
            st.write("**方法参数**")

            # 提示信息
            st.info("ℹ️ CellSAM直接在当前环境中运行。首次使用需要下载模型文件到 models/sam/ 目录")

            model_type = st.selectbox("模型类型", ["vit_b", "vit_l", "vit_h"],
                                     help="vit_b: 基础模型(91M参数), vit_l: 大模型(308M), vit_h: 超大模型(636M)")
            points_per_side = st.slider("提示点密度", 16, 64, 32, 8,
                                       help="每边生成的提示点数量。较大的值可检测更多细胞，但处理更慢。推荐32")

            # GPU选项
            if GPU_AVAILABLE and GPU_COMPATIBLE:
                use_gpu = st.checkbox(f"🚀 使用GPU加速 ({GPU_NAME})", value=True,
                                     help="CellSAM推荐使用GPU（需要4GB+ VRAM）")
            elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                use_gpu = False
                st.error(f"⚠️ GPU不兼容: {GPU_WARNING}")
                st.info("💡 将使用CPU模式处理（速度较慢）")
            else:
                use_gpu = False
                st.info("ℹ️ GPU不可用，将使用CPU处理")

            params = {"model_type": model_type, "points_per_side": points_per_side, "use_gpu": use_gpu}
        else:
            params = {}

        segment_btn = st.button("🚀 开始分割", type="primary", use_container_width=True)

    with col_right:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            st.subheader("📷 原始图像")
            st.image(image, width=400)
            st.caption(f"尺寸: {image_np.shape}")

            if segment_btn:
                st.subheader("📊 分割结果")

                # 根据方法显示不同的进度提示
                spinner_text = "正在处理..."
                if not comparison_mode and method == "Cellpose深度学习":
                    spinner_text = "🧠 正在使用Cellpose深度学习模型处理，请稍候..."

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
                        if method == "Cellpose深度学习":
                            cellpose_progress = st.progress(0)
                            st.caption("🧠 Cellpose处理进度")
                            params['progress_bar'] = cellpose_progress

                        result = segment_single_image(image_np, method, params, preprocess_options, postprocess_options)

                        # 清除Cellpose进度条
                        if method == "Cellpose深度学习":
                            cellpose_progress.empty()

                        # 显示结果
                        tab_mask, tab_overlay = st.tabs(["分割掩码", "叠加显示"])

                        with tab_mask:
                            st.image(result['mask'], use_container_width=True)

                        with tab_overlay:
                            st.image(result['overlay'], use_container_width=True)

                        # 统计信息
                        st.success("✅ 分割完成！")

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("前景像素", f"{result['foreground_pixels']:,}")
                        with col_b:
                            st.metric("前景比例", f"{result['foreground_ratio']:.2f}%")
                        with col_c:
                            st.metric("处理时间", f"{result['processing_time']*1000:.2f} ms")

                        if result['num_regions'] is not None:
                            st.info(f"🔍 检测到 {result['num_regions']} 个细胞区域")

                        # 细胞形态学特征提取
                        if (extract_morphology or use_advanced_features) and result['num_regions'] is not None and result['num_regions'] > 0:
                            st.subheader("📊 细胞形态学特征分析")

                            with st.spinner("正在提取细胞特征..."):
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
                                    st.success("✅ 高级特征提取完成！")
                                else:
                                    # 使用基础特征提取
                                    features_df = extract_cell_features(result['labeled_mask'], pixel_size=pixel_size, min_area=min_area)

                                if not features_df.empty:
                                    # 显示特征统计
                                    st.write("**特征统计摘要**")
                                    stats = get_feature_statistics(features_df)

                                    # 显示关键特征的统计信息
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("平均面积", f"{stats['area_um2']['mean']:.1f} μm²")
                                    with col2:
                                        st.metric("平均圆度", f"{stats['circularity']['mean']:.3f}")
                                    with col3:
                                        st.metric("平均长轴", f"{stats['major_axis_length']['mean']:.1f} μm")
                                    with col4:
                                        st.metric("平均短轴", f"{stats['minor_axis_length']['mean']:.1f} μm")

                                    # 显示详细特征表格
                                    with st.expander("📋 查看详细特征数据", expanded=False):
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
                                            st.caption("💡 **提示**：表格支持横向滚动，可以拖动查看所有列")
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
                                    "📊 下载特征数据",
                                    data=csv_data,
                                    file_name=f"cell_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )

                        # 单个细胞提取结果
                        if result['individual_cells'] is not None and len(result['individual_cells']) > 0:
                            st.subheader("🧬 单个细胞样本")
                            st.info(f"✅ 成功提取 {len(result['individual_cells'])} 个细胞样本")

                            # 显示前几个细胞样本
                            st.write("**样本预览（前6个）：**")
                            cols_preview = st.columns(6)
                            for idx, cell_data in enumerate(result['individual_cells'][:6]):
                                with cols_preview[idx]:
                                    st.image(cell_data['image'], caption=f"细胞 {idx+1}", use_container_width=True)

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
                                "📦 下载所有细胞样本(ZIP)",
                                data=cells_zip_buffer.getvalue(),
                                file_name=f"cells_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"❌ 分割失败: {str(e)}")
        else:
            st.info("👈 请在左侧上传细胞图像")

# ==================== 标签页2: 对比模式 ====================
with tab2:
    st.header("🔍 对比模式")
    st.caption("同时使用多种方法进行分割对比，快速评估不同算法的效果")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("⚙️ 设置")

        # 图像上传
        uploaded_file = st.file_uploader(
            "上传细胞图像",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="comparison_upload"
        )

        # 像素大小设置
        st.write("**📏 像素大小设置**")
        pixel_size = st.number_input(
            "像素大小 (μm/pixel)",
            min_value=0.01,
            max_value=10.0,
            value=0.65,
            step=0.01,
            help="输入显微镜的像素大小，用于计算实际物理尺寸",
            key="comparison_pixel_size"
        )
        st.session_state['pixel_size'] = pixel_size

        # 预处理选项
        with st.expander("🔧 预处理选项", expanded=False):
            denoise = st.checkbox("去噪处理", value=False, help="使用高斯滤波去除噪声", key="comparison_denoise")
            enhance = st.checkbox("对比度增强", value=False, help="使用CLAHE增强对比度", key="comparison_enhance")
            normalize = st.checkbox("归一化", value=False, help="归一化像素值到0-255", key="comparison_normalize")

        # 后处理选项
        with st.expander("🔬 后处理选项", expanded=False):
            closing = st.checkbox("区域闭合", value=True, help="使用形态学闭运算填充细胞边界间隙", key="comparison_closing")
            if closing:
                closing_kernel_size = st.slider("闭运算核大小", 3, 15, 3, 2, help="核越大，填充的间隙越大", key="comparison_closing_size")
            else:
                closing_kernel_size = 5

            extract_cells = st.checkbox("提取单个细胞", value=False, help="提取并保存单个细胞样本", key="comparison_extract_cells")
            if extract_cells:
                min_cell_area = st.slider("最小细胞面积", 50, 500, 50, 10, help="过滤掉面积小于此值的区域", key="comparison_min_area")
            else:
                min_cell_area = 100

            extract_morphology = st.checkbox("提取形态学特征", value=False, help="提取单个细胞的几何形态学特征", key="comparison_extract_morph")

        # 对比方法选择
        st.subheader("📊 对比方法选择")
        st.write("选择要对比的方法（至少2个）：")
        comp_methods = []
        if st.checkbox("Otsu阈值", value=True, key="comp_otsu_tab2"):
            comp_methods.append("Otsu阈值")
        if st.checkbox("自适应阈值", value=True, key="comp_adaptive_tab2"):
            comp_methods.append("自适应阈值")
        if st.checkbox("分水岭算法", value=False, key="comp_watershed_tab2"):
            comp_methods.append("分水岭算法")
        if st.checkbox("Canny边缘检测", value=False, key="comp_canny_tab2"):
            comp_methods.append("Canny边缘检测")
        if st.checkbox("Cellpose深度学习", value=False, key="comp_cellpose_tab2"):
            comp_methods.append("Cellpose深度学习")
        if st.checkbox("CellViT深度学习", value=False, key="comp_cellvit_tab2"):
            comp_methods.append("CellViT深度学习")
        if st.checkbox("CellSAM深度学习", value=False, key="comp_cellsam_tab2"):
            comp_methods.append("CellSAM深度学习")

        segment_btn = st.button("🚀 开始对比", type="primary", use_container_width=True, key="comparison_segment_btn")

    with col_right:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            st.subheader("📷 原始图像")
            st.image(image, width=400)
            st.caption(f"尺寸: {image_np.shape}")

            if segment_btn:
                st.subheader("📊 对比结果")

                if len(comp_methods) < 2:
                    st.warning("⚠️ 请至少选择2种方法进行对比")
                else:
                    with st.spinner("正在处理对比..."):
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
                                "Otsu阈值": {},
                                "自适应阈值": {"block_size": 11, "C": 2},
                                "分水岭算法": {},
                                "Canny边缘检测": {"low_threshold": 50, "high_threshold": 150},
                                "Cellpose深度学习": {
                                    "model_type": "cyto2",
                                    "diameter": None,
                                    "use_gpu": (GPU_AVAILABLE and GPU_COMPATIBLE),
                                    "batch_size": 8,
                                    "normalize": {"tile_norm_blocksize": 0}
                                },
                                "CellViT深度学习": {
                                    "model_type": "CellViT-256",
                                    "target_size": 768,
                                    "use_gpu": (GPU_AVAILABLE and GPU_COMPATIBLE)
                                },
                                "CellSAM深度学习": {
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
                                    st.metric("前景比例", f"{result['foreground_ratio']:.2f}%")
                                    st.metric("处理时间", f"{result['processing_time']*1000:.2f} ms")

                            # 性能对比表
                            st.subheader("📈 性能对比")

                            comp_df_data = []
                            for method_name in comp_methods:
                                result = comparison_results[method_name]
                                comp_df_data.append({
                                    '方法': method_name,
                                    '前景像素': result['foreground_pixels'],
                                    '前景比例(%)': f"{result['foreground_ratio']:.2f}",
                                    '处理时间(ms)': f"{result['processing_time']*1000:.2f}"
                                })

                            comp_df = pd.DataFrame(comp_df_data)
                            st.dataframe(comp_df, use_container_width=True)

                            # 推荐最快的方法
                            fastest_method = min(comp_methods, key=lambda m: comparison_results[m]['processing_time'])
                            st.success(f"⚡ 最快方法: {fastest_method} ({comparison_results[fastest_method]['processing_time']*1000:.2f} ms)")

                            # 对比结果导出
                            st.subheader("💾 导出对比结果")

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
                                    "📦 导出所有掩码(ZIP)",
                                    data=comp_zip_buffer.getvalue(),
                                    file_name=f"comparison_masks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )

                            with col_comp_export2:
                                st.download_button(
                                    "🖼️ 导出所有叠加图(ZIP)",
                                    data=comp_overlay_zip_buffer.getvalue(),
                                    file_name=f"comparison_overlays_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )

                            with col_comp_export3:
                                st.download_button(
                                    "📊 导出对比报告(CSV)",
                                    data=comp_csv_buffer.getvalue(),
                                    file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )

                        except Exception as e:
                            st.error(f"❌ 对比失败: {str(e)}")
        else:
            st.info("👈 请在左侧上传细胞图像")

# ==================== 标签页3: 模型融合 ====================
with tab3:
    st.header("🔀 多模型分割融合")
    st.markdown("融合多个深度学习模型的分割结果，提供更准确的细胞分割")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("📤 图像上传")
        fusion_uploaded = st.file_uploader(
            "上传细胞图像",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            key="fusion_upload"
        )

        if fusion_uploaded:
            fusion_image = Image.open(fusion_uploaded)
            fusion_image_np = np.array(fusion_image)

            st.subheader("🤖 模型选择")
            st.markdown("至少选择2个模型")

            # 深度学习模型
            st.markdown("**深度学习模型**")
            use_cellpose = st.checkbox("Cellpose深度学习", value=True, key="fusion_cellpose")
            use_cellvit = st.checkbox("CellViT深度学习", value=False, key="fusion_cellvit")
            use_cellsam = st.checkbox("CellSAM深度学习", value=True, key="fusion_cellsam")

            # 传统方法
            st.markdown("**传统分割方法**")
            use_watershed = st.checkbox("分水岭算法", value=False, key="fusion_watershed")
            use_otsu = st.checkbox("Otsu阈值", value=False, key="fusion_otsu")
            use_adaptive = st.checkbox("自适应阈值", value=False, key="fusion_adaptive")
            use_canny = st.checkbox("Canny边缘检测", value=False, key="fusion_canny")

            selected_models = []
            if use_cellpose: selected_models.append("cellpose")
            if use_cellvit: selected_models.append("cellvit")
            if use_cellsam: selected_models.append("cellsam")
            if use_watershed: selected_models.append("watershed")
            if use_otsu: selected_models.append("otsu")
            if use_adaptive: selected_models.append("adaptive")
            if use_canny: selected_models.append("canny")

            if len(selected_models) < 2:
                st.warning("⚠️ 请至少选择2个模型进行融合")
            else:
                # 模型参数配置
                with st.expander("⚙️ 模型参数配置", expanded=False):
                    model_params = {}

                    if use_cellpose:
                        st.markdown("**Cellpose参数**")
                        model_params['cellpose_diameter'] = st.slider("细胞直径", 10, 100, 30, key="fusion_cp_dia")
                        model_params['cellpose_gpu'] = st.checkbox("使用GPU", value=True, key="fusion_cp_gpu")

                    if use_cellvit:
                        st.markdown("**CellViT参数**")
                        model_params['cellvit_size'] = st.selectbox("模型大小", [256, 512], index=0, key="fusion_cv_size")
                        model_params['cellvit_gpu'] = st.checkbox("使用GPU", value=True, key="fusion_cv_gpu")

                    if use_cellsam:
                        st.markdown("**CellSAM参数**")
                        model_params['cellsam_points'] = st.slider("采样点数", 16, 64, 32, key="fusion_sam_points")
                        model_params['cellsam_gpu'] = st.checkbox("使用GPU", value=True, key="fusion_sam_gpu")

                    if use_watershed:
                        st.markdown("**分水岭算法参数**")
                        model_params['watershed_min_distance'] = st.slider("最小距离", 5, 30, 10, key="fusion_ws_dist")
                        model_params['watershed_threshold'] = st.slider("阈值", 0.3, 0.9, 0.5, 0.05, key="fusion_ws_thresh")

                    if use_otsu:
                        st.markdown("**Otsu阈值参数**")
                        st.info("Otsu方法自动计算最优阈值，无需手动配置参数")

                    if use_adaptive:
                        st.markdown("**自适应阈值参数**")
                        model_params['adaptive_block_size'] = st.slider("块大小", 11, 51, 21, 2, key="fusion_adp_block")
                        model_params['adaptive_c'] = st.slider("常数C", 0, 20, 5, key="fusion_adp_c")

                    if use_canny:
                        st.markdown("**Canny边缘检测参数**")
                        model_params['canny_threshold1'] = st.slider("低阈值", 20, 150, 50, key="fusion_canny_t1")
                        model_params['canny_threshold2'] = st.slider("高阈值", 50, 300, 150, key="fusion_canny_t2")

                # 融合策略选择
                st.subheader("🎯 融合策略")

                # 策略类型选择
                strategy_type = st.radio(
                    "选择策略类型",
                    ["简单策略", "高级融合方法 (DST)"],
                    key="strategy_type"
                )

                if strategy_type == "简单策略":
                    fusion_strategy = st.radio(
                        "选择融合策略",
                        ["majority", "weighted", "union", "intersection"],
                        format_func=lambda x: {
                            "majority": "简单投票 (推荐)",
                            "weighted": "加权投票",
                            "union": "激进融合 (高召回)",
                            "intersection": "保守融合 (高精确)"
                        }[x],
                        key="fusion_strategy"
                    )
                else:
                    fusion_strategy = "dempster_shafer"
                    st.info("🎓 **Dempster-Shafer理论融合**：基于证据理论的高级融合方法，能够明确建模不确定性和量化模型冲突。")

                # 高级选项
                with st.expander("🔧 高级选项", expanded=False):
                    iou_threshold = st.slider("IoU匹配阈值", 0.2, 0.9, 0.2, 0.05, key="fusion_iou")

                    # 最小投票数设置（只在3个或更多模型时显示slider）
                    if len(selected_models) == 2:
                        st.info("💡 最小投票数: 2 (固定，因为只选择了2个模型)")
                        min_vote_count = 2
                    else:
                        min_vote_count = st.slider("最小投票数", 2, len(selected_models), 2, key="fusion_min_vote")

                    if fusion_strategy == "weighted":
                        st.markdown("**模型权重设置**")
                        weights = {}
                        if use_cellpose:
                            weights['cellpose'] = st.slider("Cellpose权重", 0.1, 2.0, 1.0, 0.1, key="w_cp")
                        if use_cellvit:
                            weights['cellvit'] = st.slider("CellViT权重", 0.1, 2.0, 1.0, 0.1, key="w_cv")
                        if use_cellsam:
                            weights['cellsam'] = st.slider("CellSAM权重", 0.1, 2.0, 1.0, 0.1, key="w_sam")
                    else:
                        weights = None

                    # DST特定参数
                    if fusion_strategy == "dempster_shafer":
                        st.markdown("**🎓 DST模型可靠性参数**")
                        st.caption("可靠性表示模型的整体置信度，范围[0,1]，越高表示越信任该模型")

                        model_reliabilities = {}
                        if use_cellpose:
                            model_reliabilities['cellpose'] = st.slider("Cellpose可靠性", 0.5, 1.0, 0.9, 0.05, key="dst_r_cp")
                        if use_cellvit:
                            model_reliabilities['cellvit'] = st.slider("CellViT可靠性", 0.5, 1.0, 0.85, 0.05, key="dst_r_cv")
                        if use_cellsam:
                            model_reliabilities['cellsam'] = st.slider("CellSAM可靠性", 0.5, 1.0, 0.8, 0.05, key="dst_r_sam")

                        # 传统方法的可靠性
                        if 'watershed' in selected_models:
                            model_reliabilities['watershed'] = st.slider("Watershed可靠性", 0.5, 1.0, 0.7, 0.05, key="dst_r_ws")
                        if 'otsu' in selected_models:
                            model_reliabilities['otsu'] = st.slider("Otsu可靠性", 0.5, 1.0, 0.65, 0.05, key="dst_r_otsu")
                        if 'adaptive' in selected_models:
                            model_reliabilities['adaptive'] = st.slider("Adaptive可靠性", 0.5, 1.0, 0.65, 0.05, key="dst_r_adp")
                        if 'canny' in selected_models:
                            model_reliabilities['canny'] = st.slider("Canny可靠性", 0.5, 1.0, 0.6, 0.05, key="dst_r_canny")

                        conflict_threshold = st.slider("冲突阈值", 0.3, 0.9, 0.4, 0.05, key="dst_conflict_th",
                                                      help="超过此阈值的实例将被标记为高冲突")
                    else:
                        model_reliabilities = None
                        conflict_threshold = 0.4

                # 开始融合按钮
                if st.button("🚀 开始融合", type="primary", key="start_fusion"):
                    run_fusion_pipeline(
                        fusion_image_np, selected_models, fusion_strategy,
                        iou_threshold, min_vote_count, weights, model_params, col_right,
                        model_reliabilities=model_reliabilities, conflict_threshold=conflict_threshold
                    )

    with col_right:
        if fusion_uploaded:
            st.subheader("📷 原始图像")
            st.image(fusion_image, width=400)
            st.caption(f"尺寸: {fusion_image_np.shape}")

            st.subheader("📊 融合结果")
            st.info("配置参数后点击'开始融合'按钮")
        else:
            st.info("👈 请在左侧上传细胞图像")

# ==================== 标签页4: 批量处理 ====================
with tab4:
    st.header("📦 批量处理")

    col_batch_left, col_batch_right = st.columns([1, 2])

    with col_batch_left:
        st.subheader("⚙️ 批量设置")

        # 批量上传
        uploaded_files = st.file_uploader(
            "上传多张细胞图像",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key="batch_upload"
        )

        # 预处理选项
        with st.expander("🔧 预处理选项"):
            batch_denoise = st.checkbox("去噪处理", value=False, key="batch_denoise")
            batch_enhance = st.checkbox("对比度增强", value=False, key="batch_enhance")
            batch_normalize = st.checkbox("归一化", value=False, key="batch_normalize")

        # 后处理选项
        with st.expander("🔬 后处理选项"):
            batch_closing = st.checkbox("区域闭合", value=True, key="batch_closing", help="使用形态学闭运算填充细胞边界间隙")
            if batch_closing:
                batch_closing_kernel_size = st.slider("闭运算核大小", 3, 15, 5, 2, key="batch_closing_kernel", help="核越大，填充的间隙越大")
            else:
                batch_closing_kernel_size = 5

            batch_extract_cells = st.checkbox("提取单个细胞", value=False, key="batch_extract_cells", help="提取并保存单个细胞样本")
            if batch_extract_cells:
                batch_min_cell_area = st.slider("最小细胞面积", 50, 500, 100, 10, key="batch_min_cell_area", help="过滤掉面积小于此值的区域")

            batch_extract_morphology = st.checkbox("提取形态学特征", value=False, key="batch_extract_morphology", help="提取单个细胞的几何形态学特征（面积、周长、圆度等）")

            # 批量高级特征提取选项
            batch_use_advanced_features = st.checkbox("使用高级特征提取", value=False, key="batch_use_advanced_features", help="提取更高级的形态学、纹理和强度特征")
            if batch_use_advanced_features:
                st.caption("**高级特征类别**")
                batch_include_hu_moments = st.checkbox("Hu矩特征", value=True, key="batch_hu_moments", help="7个旋转、缩放、平移不变的形状描述符")
                batch_include_intensity = st.checkbox("强度统计特征", value=True, key="batch_intensity", help="灰度统计（均值、标准差、偏度、峰度、熵）")
                batch_include_texture = st.checkbox("纹理特征(GLCM)", value=True, key="batch_texture", help="基于灰度共生矩阵的Haralick纹理特征")
                batch_include_boundary = st.checkbox("边界复杂度特征", value=True, key="batch_boundary", help="边界粗糙度、凹凸性分析")
                batch_include_advanced_shape = st.checkbox("高级形状特征", value=True, key="batch_advanced_shape", help="椭圆度、伸长度、分形维数等")

        # 分割方法
        batch_method = st.selectbox(
            "分割方法",
            ["Otsu阈值", "自适应阈值", "分水岭算法", "Canny边缘检测", "Cellpose深度学习", "CellViT深度学习", "CellSAM深度学习"],
            key="batch_method"
        )

        # 方法参数
        with st.expander("📐 方法参数"):
            if batch_method == "自适应阈值":
                batch_block_size = st.slider("块大小", 3, 51, 11, 2, key="batch_block_size")
                batch_C = st.slider("常数C", 0, 20, 2, key="batch_C")
                batch_params = {"block_size": batch_block_size, "C": batch_C}
            elif batch_method == "Canny边缘检测":
                batch_low = st.slider("低阈值", 0, 200, 50, 10, key="batch_low")
                batch_high = st.slider("高阈值", 0, 300, 150, 10, key="batch_high")
                batch_params = {"low_threshold": batch_low, "high_threshold": batch_high}
            elif batch_method == "Cellpose深度学习":
                batch_model_type = st.selectbox("模型类型", ["cyto2", "cyto", "nuclei"], key="batch_model_type",
                                               help="cyto2: 细胞质模型(推荐), cyto: 旧版细胞质模型, nuclei: 细胞核模型")
                batch_diameter = st.slider("细胞直径(像素)", 0, 100, 30, 5, key="batch_diameter",
                                          help="设置为0则自动检测")
                if batch_diameter == 0:
                    batch_diameter = None

                # GPU选项
                if GPU_AVAILABLE and GPU_COMPATIBLE:
                    batch_use_gpu = st.checkbox(f"🚀 使用GPU加速 ({GPU_NAME})", value=True, key="batch_use_gpu",
                                               help="启用GPU可大幅提升处理速度（10-50倍）")
                elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                    batch_use_gpu = False
                    st.error(f"⚠️ GPU不兼容: {GPU_WARNING}")
                    st.info("💡 将使用CPU模式处理（速度较慢但稳定）")
                else:
                    batch_use_gpu = False
                    st.info("ℹ️ GPU不可用，将使用CPU处理")

                batch_params = {"model_type": batch_model_type, "diameter": batch_diameter, "use_gpu": batch_use_gpu}
            elif batch_method == "CellViT深度学习":
                # 环境检查
                if not CELLVIT_ENV_OK:
                    st.error(f"⚠️ CellViT专用环境未找到！")
                    st.warning("请先创建CellViT环境")
                else:
                    st.success(f"✅ CellViT专用环境已就绪")

                batch_model_type = st.selectbox("模型类型", ["CellViT-256"], key="batch_cellvit_model",
                                               help="CellViT-256: 基于Vision Transformer的细胞核分割模型")
                batch_target_size = st.slider("目标图像大小", 256, 1024, 512, 64, key="batch_target_size",
                                             help="图像会被调整到此大小进行处理。较大的值可检测更多小细胞，但处理更慢。推荐512-768")

                # GPU选项
                if GPU_AVAILABLE and GPU_COMPATIBLE:
                    batch_use_gpu = st.checkbox(f"🚀 使用GPU加速 ({GPU_NAME})", value=True, key="batch_cellvit_gpu",
                                               help="CellViT推荐使用GPU")
                elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                    batch_use_gpu = False
                    st.error(f"⚠️ GPU不兼容: {GPU_WARNING}")
                else:
                    batch_use_gpu = False
                    st.info("ℹ️ GPU不可用，将使用CPU处理")

                batch_params = {"model_type": batch_model_type, "target_size": batch_target_size, "use_gpu": batch_use_gpu}
            elif batch_method == "CellSAM深度学习":
                # 提示信息
                st.info("ℹ️ CellSAM直接在当前环境中运行。首次使用需要下载模型文件到 models/sam/ 目录")

                batch_model_type = st.selectbox("模型类型", ["vit_b", "vit_l", "vit_h"], key="batch_cellsam_model",
                                               help="vit_b: 基础模型(91M参数), vit_l: 大模型(308M), vit_h: 超大模型(636M)")
                batch_points_per_side = st.slider("提示点密度", 16, 64, 32, 8, key="batch_points_per_side",
                                                 help="每边生成的提示点数量。较大的值可检测更多细胞，但处理更慢。推荐32")

                # GPU选项
                if GPU_AVAILABLE and GPU_COMPATIBLE:
                    batch_use_gpu = st.checkbox(f"🚀 使用GPU加速 ({GPU_NAME})", value=True, key="batch_cellsam_gpu",
                                               help="CellSAM推荐使用GPU")
                elif GPU_AVAILABLE and not GPU_COMPATIBLE:
                    batch_use_gpu = False
                    st.error(f"⚠️ GPU不兼容: {GPU_WARNING}")
                else:
                    batch_use_gpu = False
                    st.info("ℹ️ GPU不可用，将使用CPU处理")

                batch_params = {"model_type": batch_model_type, "points_per_side": batch_points_per_side, "use_gpu": batch_use_gpu}
            else:
                batch_params = {}

        batch_process_btn = st.button("🚀 批量处理", type="primary", use_container_width=True)

    with col_batch_right:
        if uploaded_files:
            st.info(f"📁 已上传 {len(uploaded_files)} 张图像")

            if batch_process_btn:
                st.subheader("📊 处理进度")

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
                        st.warning(f"⚠️ {uploaded_file.name} 读取失败: {str(e)}")

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
                                st.warning(f"⚠️ {result_dict['filename']} 处理失败: {result_dict['error']}")

                        except Exception as e:
                            st.warning(f"⚠️ {filename} 处理异常: {str(e)}")

                        # 更新进度
                        progress_bar.progress(completed_count / len(task_args))
                        status_text.text(f"⚙️ 已完成 {completed_count}/{len(task_args)} 张图像...")

                status_text.text("✅ 批量处理完成！")
                st.session_state.batch_results = batch_results

                # 显示统计摘要
                st.subheader("📈 统计摘要")

                if batch_results:
                    df_data = []
                    for item in batch_results:
                        df_data.append({
                            '文件名': item['filename'],
                            '前景像素': item['result']['foreground_pixels'],
                            '前景比例(%)': f"{item['result']['foreground_ratio']:.2f}",
                            '处理时间(ms)': f"{item['result']['processing_time']*1000:.2f}"
                        })

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)

                    # 可视化结果
                    st.subheader("🖼️ 可视化结果")

                    # 使用expander来展示每张图片的结果
                    for idx, item in enumerate(batch_results):
                        with st.expander(f"📷 {item['filename']}", expanded=(idx == 0)):
                            col_vis1, col_vis2, col_vis3 = st.columns(3)

                            with col_vis1:
                                st.write("**原图**")
                                st.image(item['image'], use_container_width=True)

                            with col_vis2:
                                st.write("**分割掩码**")
                                st.image(item['result']['mask'], use_container_width=True)

                            with col_vis3:
                                st.write("**叠加显示**")
                                st.image(item['result']['overlay'], use_container_width=True)

                            # 显示统计信息
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("前景像素", f"{item['result']['foreground_pixels']:,}")
                            with col_stat2:
                                st.metric("前景比例", f"{item['result']['foreground_ratio']:.2f}%")
                            with col_stat3:
                                st.metric("处理时间", f"{item['result']['processing_time']*1000:.2f} ms")

                    # 批量导出
                    st.subheader("💾 批量导出")

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
                            "📦 导出所有掩码(ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name=f"masks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

                    with col_export_all2:
                        st.download_button(
                            "🖼️ 导出所有叠加图(ZIP)",
                            data=overlay_zip_buffer.getvalue(),
                            file_name=f"overlays_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

                    with col_export_all3:
                        st.download_button(
                            "📊 导出统计报告(CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    # 单个细胞批量导出
                    total_cells = sum(len(item['result']['individual_cells']) if item['result']['individual_cells'] else 0 for item in batch_results)
                    if total_cells > 0:
                        st.write("")  # 添加间距
                        st.info(f"🧬 共提取 {total_cells} 个细胞样本")

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
                            "🧬 导出所有细胞样本(ZIP)",
                            data=all_cells_zip_buffer.getvalue(),
                            file_name=f"all_cells_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

        else:
            st.info("👈 请在左侧上传多张细胞图像")

# ==================== 标签页6: 异常检测 ====================
with tab6:
    st.header("🔍 异常检测")
    st.caption("上传细胞特征CSV文件进行异常检测，识别形态学参数异常的细胞样本")
    st.info("💡 使用说明请查看页面顶部的 📖 使用说明")

    st.markdown("---")

    # CSV文件上传
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded_csv = st.file_uploader(
            "📁 上传细胞特征CSV文件",
            type=["csv"],
            key="ml_csv_upload",
            help="上传从图像分割模块导出的细胞特征CSV文件"
        )

    with col_info:
        if uploaded_csv is not None:
            st.success("✅ 文件已上传")
            st.info(f"文件名: {uploaded_csv.name}")

    # 处理上传的CSV文件
    if uploaded_csv is not None:
        try:
            # 读取CSV文件
            ml_features_df = pd.read_csv(uploaded_csv)

            # 验证CSV格式
            required_cols = ['area_um2', 'perimeter_um', 'circularity']
            missing_cols = [col for col in required_cols if col not in ml_features_df.columns]

            if missing_cols:
                st.error(f"❌ CSV文件格式不正确！缺少必需的列: {', '.join(missing_cols)}")
                st.info("请确保CSV文件是从图像分割模块导出的细胞特征文件")
            else:
                # 显示数据概览
                st.success(f"✅ CSV文件加载成功！共 {len(ml_features_df)} 个细胞")

                # 数据预览
                with st.expander("📊 数据预览", expanded=False):
                    st.write(f"**数据维度**: {ml_features_df.shape[0]} 行 × {ml_features_df.shape[1]} 列")
                    st.write("**前10行数据**:")
                    st.dataframe(ml_features_df.head(10), use_container_width=True)

                    # 显示特征列表
                    feature_cols = [col for col in ml_features_df.columns if col not in
                                   ['sequential_id', 'cell_id', 'centroid_x', 'centroid_y',
                                    'bbox_min_row', 'bbox_min_col', 'bbox_max_row', 'bbox_max_col']]
                    st.write(f"**可用特征** ({len(feature_cols)} 个):")
                    st.write(", ".join(feature_cols))

                st.markdown("---")

                # 保存到session_state
                st.session_state['ml_features_df'] = ml_features_df

        except Exception as e:
            st.error(f"❌ 读取CSV文件失败: {str(e)}")
            st.info("请确保上传的是有效的CSV文件")

    # 机器学习异常识别UI（只在有数据时显示）
    if 'ml_features_df' in st.session_state and not st.session_state['ml_features_df'].empty:
        ml_features_df = st.session_state['ml_features_df']

        st.markdown("---")

        st.subheader("🔍 异常检测")
        st.caption("使用无监督学习算法识别形态学参数异常的细胞样本")

        # 异常检测算法选择
        col_anomaly_algo, col_anomaly_param = st.columns([1, 2])

        with col_anomaly_algo:
            ml_anomaly_method = st.selectbox(
                "异常检测算法",
                ["Isolation Forest", "LOF", "One-Class SVM", "Elliptic Envelope"],
                key="ml_anomaly_method",
                help="选择异常检测算法。Isolation Forest适合高维数据，LOF基于密度，One-Class SVM学习正常边界，Elliptic Envelope假设高斯分布"
            )

        with col_anomaly_param:
            # 根据选择的算法显示不同的参数控件
            if ml_anomaly_method == "Isolation Forest":
                ml_contamination = st.slider(
                    "异常比例(contamination)",
                    0.01, 0.5, 0.1, 0.01,
                    key="ml_contamination_if",
                    help="预期异常样本的比例"
                )

            elif ml_anomaly_method == "LOF":
                col_cont, col_neigh = st.columns(2)
                with col_cont:
                    ml_contamination = st.slider(
                        "异常比例",
                        0.01, 0.5, 0.1, 0.01,
                        key="ml_contamination_lof"
                    )
                with col_neigh:
                    ml_n_neighbors_lof = st.slider(
                        "邻居数量",
                        5, 50, 20, 5,
                        key="ml_n_neighbors_lof",
                        help="用于计算局部密度的邻居数量"
                    )

            elif ml_anomaly_method == "One-Class SVM":
                col_nu, col_kernel = st.columns(2)
                with col_nu:
                    ml_nu = st.slider(
                        "异常上界(nu)",
                        0.01, 0.5, 0.1, 0.01,
                        key="ml_nu",
                        help="异常样本比例的上界"
                    )
                with col_kernel:
                    ml_kernel = st.selectbox(
                        "核函数",
                        ["rbf", "linear", "poly", "sigmoid"],
                        key="ml_kernel"
                    )

            elif ml_anomaly_method == "Elliptic Envelope":
                ml_contamination = st.slider(
                    "异常比例(contamination)",
                    0.01, 0.5, 0.1, 0.01,
                    key="ml_contamination_ee",
                    help="预期异常样本的比例"
                )

        # 执行异常检测按钮
        if st.button("🚀 执行异常检测", type="primary", use_container_width=True, key="ml_anomaly_button"):
            with st.spinner("正在执行异常检测..."):
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

                    st.success(f"✅ 异常检测完成！")

                    # 显示异常检测结果
                    st.write("**异常检测结果**")
                    col_r1, col_r2, col_r3 = st.columns(3)

                    with col_r1:
                        st.metric("正常样本", f"{info['n_normal']} 个",
                                 help="被识别为正常的细胞数量")
                    with col_r2:
                        st.metric("异常样本", f"{info['n_anomalies']} 个",
                                 help="被识别为异常的细胞数量")
                    with col_r3:
                        st.metric("异常比例", f"{info['anomaly_ratio']*100:.1f}%",
                                 help="异常样本占总样本的比例")

                except Exception as e:
                    st.error(f"❌ 异常检测失败: {str(e)}")

        # 异常检测可视化和统计（只在有异常检测结果时显示）
        if 'ml_anomaly_labels' in st.session_state and 'ml_features_df' in st.session_state:
            st.markdown("---")
            with st.expander("📊 异常检测可视化与统计", expanded=True):
                st.caption("可视化异常样本分布并分析异常特征")

                # 异常统计分析
                st.write("**异常特征统计**")
                st.caption("对比正常样本和异常样本的特征差异")

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

                    st.caption("💡 difference列表示正常样本和异常样本在该特征上的均值差异，值越大表示该特征越能区分正常和异常样本")

                except Exception as e:
                    st.error(f"❌ 统计分析失败: {str(e)}")

            st.markdown("---")

            # 算法可视化
            st.write("**算法可视化**")
            st.caption("展示异常检测算法的工作原理和检测结果")

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
                    st.caption("💡 左上：样本分布（PCA降维）| 右上：异常分数分布 | 左下：异常分数热图 | 右下：统计信息")
                    st.info(f"🎨 当前使用SVG矢量图显示（{n_samples}个样本），支持无限放大而不失真")
                else:
                    # 数据点多于10000，使用PNG栅格图（避免浏览器卡顿）
                    st.pyplot(fig)
                    st.caption("💡 左上：样本分布（PCA降维）| 右上：异常分数分布 | 左下：异常分数热图 | 右下：统计信息")
                    st.info(f"📊 当前使用PNG栅格图显示（{n_samples}个样本），避免浏览器卡顿。下载SVG格式可获得矢量图")

                # 下载按钮
                st.write("**下载可视化图片**")
                col_dl1, col_dl2, col_dl3 = st.columns(3)

                with col_dl1:
                    # PNG格式
                    png_buffer = io.BytesIO()
                    fig.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
                    png_buffer.seek(0)
                    st.download_button(
                        "📥 下载PNG",
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
                        "📥 下载SVG",
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
                        "📥 下载PDF",
                        data=pdf_buffer,
                        file_name=f"anomaly_{method.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ 算法可视化失败: {str(e)}")

            st.markdown("---")

            # 降维可视化
            st.write("**异常样本可视化**")
            col_viz_method_anomaly, col_viz_param_anomaly = st.columns([1, 2])

            with col_viz_method_anomaly:
                ml_viz_method_anomaly = st.selectbox(
                    "降维方法",
                    ["PCA", "t-SNE", "UMAP"],
                    key="ml_viz_method_anomaly",
                    help="使用降维方法将高维特征投影到2D空间"
                )

            with col_viz_param_anomaly:
                if ml_viz_method_anomaly == "t-SNE":
                    ml_perplexity_anomaly = st.slider(
                        "困惑度(perplexity)",
                        5, 50, 30, 5,
                        key="ml_perplexity_anomaly"
                    )
                elif ml_viz_method_anomaly == "UMAP":
                    col_n, col_d = st.columns(2)
                    with col_n:
                        ml_n_neighbors_anomaly = st.slider(
                            "邻居数量",
                            5, 50, 15, 5,
                            key="ml_n_neighbors_anomaly"
                        )
                    with col_d:
                        ml_min_dist_anomaly = st.slider(
                            "最小距离",
                            0.0, 0.99, 0.1, 0.05,
                            key="ml_min_dist_anomaly"
                        )

            # 执行可视化
            if st.button("🎨 生成异常检测可视化", type="secondary", use_container_width=True, key="ml_viz_anomaly_button"):
                with st.spinner(f"正在执行{ml_viz_method_anomaly}降维..."):
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
                        viz_df_anomaly['status'] = viz_df_anomaly['anomaly_label'].map({1: '正常', -1: '异常'})

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
                            title=f'异常检测可视化 ({ml_viz_method_anomaly})',
                            labels={'component_1': f'{ml_viz_method_anomaly}1',
                                   'component_2': f'{ml_viz_method_anomaly}2',
                                   'status': '样本状态'},
                            color_discrete_map={'正常': '#2ecc71', '异常': '#e74c3c'}
                        )

                        fig.update_traces(marker=dict(size=8, opacity=0.7))
                        fig.update_layout(height=600)

                        st.plotly_chart(fig, use_container_width=True)

                        st.success(f"✅ {ml_viz_method_anomaly}可视化完成！")

                        # 下载按钮
                        st.write("**下载可视化图片**")
                        col_dl_plotly1, col_dl_plotly2, col_dl_plotly3 = st.columns(3)

                        with col_dl_plotly1:
                            # HTML格式（交互式）
                            html_buffer = io.StringIO()
                            fig.write_html(html_buffer)
                            html_str = html_buffer.getvalue()
                            st.download_button(
                                "📥 下载HTML（交互式）",
                                data=html_str,
                                file_name=f"anomaly_viz_{ml_viz_method_anomaly}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html",
                                use_container_width=True,
                                help="下载交互式HTML文件，可在浏览器中打开"
                            )

                        with col_dl_plotly2:
                            # PNG格式
                            try:
                                png_bytes = fig.to_image(format="png", width=1200, height=800)
                                st.download_button(
                                    "📥 下载PNG",
                                    data=png_bytes,
                                    file_name=f"anomaly_viz_{ml_viz_method_anomaly}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.caption("⚠️ PNG导出需要安装kaleido包")

                        with col_dl_plotly3:
                            # SVG格式
                            try:
                                svg_bytes = fig.to_image(format="svg", width=1200, height=800)
                                st.download_button(
                                    "📥 下载SVG",
                                    data=svg_bytes,
                                    file_name=f"anomaly_viz_{ml_viz_method_anomaly}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                                    mime="image/svg+xml",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.caption("⚠️ SVG导出需要安装kaleido包")

                    except Exception as e:
                        st.error(f"❌ 可视化失败: {str(e)}")

    # 结果导出section（只在有聚类或异常检测结果时显示）
    if 'ml_features_df_clustered' in st.session_state or 'ml_features_df_anomaly' in st.session_state:
        st.markdown("---")
        st.subheader("💾 导出结果")

        # 聚类结果导出
        if 'ml_features_df_clustered' in st.session_state:
            ml_features_df_clustered = st.session_state['ml_features_df_clustered']

            col_export1, col_export2 = st.columns(2)

            with col_export1:
                # 导出带聚类标签的CSV
                csv_data = ml_features_df_clustered.to_csv(index=False)
                st.download_button(
                    "📥 下载带聚类标签的CSV",
                    data=csv_data,
                    file_name=f"clustered_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="下载包含聚类标签的完整特征数据"
                )

            with col_export2:
                # 显示数据预览
                st.info(f"📊 聚类数据: {len(ml_features_df_clustered)} 个细胞，{len(ml_features_df_clustered.columns)} 个特征列")

        # 异常检测结果导出
        if 'ml_features_df_anomaly' in st.session_state:
            ml_features_df_anomaly = st.session_state['ml_features_df_anomaly']

            col_export3, col_export4 = st.columns(2)

            with col_export3:
                # 导出带异常标签的CSV
                csv_data_anomaly = ml_features_df_anomaly.to_csv(index=False)
                st.download_button(
                    "📥 下载带异常标签的CSV",
                    data=csv_data_anomaly,
                    file_name=f"anomaly_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="下载包含异常检测标签的完整特征数据"
                )

            with col_export4:
                # 显示数据预览
                n_normal = (ml_features_df_anomaly['anomaly'] == 1).sum()
                n_anomaly = (ml_features_df_anomaly['anomaly'] == -1).sum()
                st.info(f"📊 异常检测数据: {len(ml_features_df_anomaly)} 个细胞 (正常: {n_normal}, 异常: {n_anomaly})")

            # 导出仅正常样本的CSV
            st.write("")
            col_export5, col_export6 = st.columns(2)

            with col_export5:
                # 导出仅正常样本
                ml_features_df_normal = ml_features_df_anomaly[ml_features_df_anomaly['anomaly'] == 1].copy()
                csv_data_normal = ml_features_df_normal.to_csv(index=False)
                st.download_button(
                    "✅ 下载仅正常样本的CSV",
                    data=csv_data_normal,
                    file_name=f"normal_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="下载仅包含正常样本的特征数据（排除异常样本）"
                )

            with col_export6:
                st.caption(f"💡 正常样本数据包含 {len(ml_features_df_normal)} 个细胞（已排除 {n_anomaly} 个异常样本）")

    else:
        st.info("👆 请上传细胞特征CSV文件开始分析")

# ==================== 标签页7: 聚类分析 ====================
with tab7:
    st.header("📊 聚类分析")
    st.caption("上传细胞特征CSV文件进行聚类分析，自动发现细胞亚群")
    st.info("💡 使用说明请查看页面顶部的 📖 使用说明")

    st.markdown("---")

    # CSV文件上传
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded_csv = st.file_uploader(
            "上传细胞特征CSV文件",
            type=['csv'],
            key="clustering_csv_upload",
            help="上传包含细胞形态学特征的CSV文件"
        )

    with col_info:
        st.info("📋 CSV文件应包含细胞的形态学特征（面积、圆度、长轴、短轴等）")

    # 读取CSV文件
    if uploaded_csv is not None:
        try:
            import pandas as pd
            clustering_features_df = pd.read_csv(uploaded_csv)

            st.success(f"✅ 成功读取CSV文件！共 {len(clustering_features_df)} 个细胞样本")

            # 显示数据预览
            with st.expander("📊 数据预览", expanded=False):
                st.write(f"**数据维度**: {clustering_features_df.shape[0]} 行 × {clustering_features_df.shape[1]} 列")
                st.dataframe(clustering_features_df.head(10), use_container_width=True)

                # 显示特征列表
                st.write("**特征列表**:")
                feature_cols = [col for col in clustering_features_df.columns if col not in ['cell_id', 'sequential_id', 'image_name']]
                st.write(", ".join(feature_cols))

            # 保存到session_state
            st.session_state['clustering_features_df'] = clustering_features_df

        except Exception as e:
            st.error(f"❌ 读取CSV文件失败: {str(e)}")
            st.info("请确保上传的是有效的CSV文件")

    # 聚类分析UI（只在有数据时显示）
    if 'clustering_features_df' in st.session_state and not st.session_state['clustering_features_df'].empty:
        clustering_features_df = st.session_state['clustering_features_df']

        st.markdown("---")
        st.subheader("📊 聚类分析")

        # Left-right column layout for clustering
        col_left, col_right = st.columns([2, 3])

        with col_left:
            st.write("**聚类参数设置**")

            # Clustering algorithm selection
            clustering_method = st.selectbox(
                "聚类算法",
                ["K-means", "DBSCAN", "层次聚类", "GMM"],
                key="clustering_method",
                help="选择聚类算法。K-means适合球形簇，DBSCAN适合任意形状，层次聚类生成树状图，GMM是概率聚类"
            )

            # Algorithm-specific parameters
            if clustering_method == "K-means":
                col_k, col_auto = st.columns(2)
                with col_k:
                    n_clusters = st.number_input("聚类数量", min_value=2, max_value=10, value=3, step=1, key="clustering_n_clusters")
                with col_auto:
                    auto_k = st.checkbox("自动检测最佳k值", value=False, key="clustering_auto_k", help="使用轮廓系数自动寻找最佳聚类数量")

            elif clustering_method == "DBSCAN":
                col_eps, col_min = st.columns(2)
                with col_eps:
                    eps = st.number_input("邻域半径(eps)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="clustering_eps")
                    auto_eps = st.checkbox("自动估计eps", value=True, key="clustering_auto_eps")
                with col_min:
                    min_samples = st.number_input("最小样本数", min_value=2, max_value=20, value=5, step=1, key="clustering_min_samples")

            elif clustering_method == "层次聚类":
                col_k, col_link = st.columns(2)
                with col_k:
                    n_clusters = st.number_input("聚类数量", min_value=2, max_value=10, value=3, step=1, key="clustering_n_clusters_hier")
                with col_link:
                    linkage = st.selectbox("链接方法", ["ward", "complete", "average", "single"], key="clustering_linkage")

            elif clustering_method == "GMM":
                n_components = st.number_input("高斯分量数", min_value=2, max_value=10, value=3, step=1, key="clustering_n_components")

            # Execute clustering button
            if st.button("🚀 执行聚类分析", type="primary", use_container_width=True, key="clustering_execute_button"):
                with st.spinner("正在执行聚类分析..."):
                    try:
                        # Execute clustering
                        if clustering_method == "K-means":
                            if auto_k:
                                optimal_results = find_optimal_clusters(clustering_features_df, max_k=10, method='kmeans')
                                n_clusters = optimal_results['best_k']
                                st.info(f"🎯 自动检测到最佳聚类数量: k={n_clusters}")
                            labels, info = perform_kmeans(clustering_features_df, n_clusters=n_clusters)

                        elif clustering_method == "DBSCAN":
                            eps_val = None if auto_eps else eps
                            labels, info = perform_dbscan(clustering_features_df, eps=eps_val, min_samples=min_samples)

                        elif clustering_method == "层次聚类":
                            labels, info = perform_hierarchical(clustering_features_df, n_clusters=n_clusters, linkage=linkage)

                        elif clustering_method == "GMM":
                            labels, info = perform_gmm(clustering_features_df, n_components=n_components)

                        # Save clustering results to session_state
                        st.session_state['clustering_labels'] = labels
                        st.session_state['clustering_info'] = info

                        # Add clustering labels to features_df
                        clustering_features_df_clustered = clustering_features_df.copy()
                        clustering_features_df_clustered['cluster'] = labels
                        st.session_state['clustering_features_df_clustered'] = clustering_features_df_clustered

                        st.success(f"✅ 聚类分析完成！")

                    except Exception as e:
                        st.error(f"❌ 聚类分析失败: {str(e)}")

            # Display clustering results (only when results exist)
            if 'clustering_labels' in st.session_state and 'clustering_info' in st.session_state:
                st.markdown("---")
                st.write("**聚类质量指标**")

                info = st.session_state['clustering_info']
                labels = st.session_state['clustering_labels']

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("轮廓系数", f"{info['silhouette_score']:.3f}", help="范围[-1,1]，越接近1表示聚类质量越好")
                with col_m2:
                    st.metric("Davies-Bouldin指数", f"{info['davies_bouldin_score']:.3f}", help="越小越好，表示簇间分离度")

                st.write("**聚类统计**")
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                st.write(f"- 发现 **{n_clusters_found}** 个聚类")

                if -1 in labels:
                    n_noise = list(labels).count(-1)
                    st.write(f"- 噪声点: **{n_noise}** 个")

                cluster_counts = pd.Series(labels).value_counts().sort_index()
                for cluster_id, count in cluster_counts.items():
                    if cluster_id != -1:
                        st.write(f"- 聚类 {cluster_id}: **{count}** 个细胞")

        with col_right:
            st.write("**聚类可视化**")

            # Visualization only when clustering results exist
            if 'clustering_labels' in st.session_state and 'clustering_features_df' in st.session_state:

                # Dimensionality reduction method selection
                viz_method = st.selectbox(
                    "降维方法",
                    ["PCA", "t-SNE", "UMAP"],
                    key="clustering_viz_method",
                    help="PCA: 快速线性降维; t-SNE: 保留局部结构; UMAP: 保留全局和局部结构"
                )

                # Method-specific parameters
                if viz_method == "t-SNE":
                    col_perp, col_iter = st.columns(2)
                    with col_perp:
                        perplexity = st.slider("困惑度", 5, 50, 30, 5, key="clustering_perplexity")
                    with col_iter:
                        n_iter = st.slider("迭代次数", 250, 2000, 1000, 250, key="clustering_n_iter")
                elif viz_method == "UMAP":
                    col_neighbors, col_dist = st.columns(2)
                    with col_neighbors:
                        n_neighbors = st.slider("邻居数量", 5, 50, 15, 5, key="clustering_n_neighbors")
                    with col_dist:
                        min_dist = st.slider("最小距离", 0.0, 0.99, 0.1, 0.05, key="clustering_min_dist")

                # Execute visualization
                if st.button("🎨 生成可视化", type="secondary", use_container_width=True, key="clustering_viz_button"):
                    with st.spinner(f"正在执行{viz_method}降维..."):
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
                                title=f'细胞聚类可视化 ({viz_method})',
                                labels={'component_1': f'{viz_method}1', 'component_2': f'{viz_method}2', 'cluster': '聚类'},
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )

                            fig.update_traces(marker=dict(size=8, opacity=0.7))
                            fig.update_layout(height=500)

                            # Save to session_state for display outside button block
                            st.session_state['clustering_viz_fig'] = fig

                            st.success(f"✅ {viz_method}可视化完成！")

                        except Exception as e:
                            st.error(f"❌ 可视化失败: {str(e)}")

                # Display saved visualization
                if 'clustering_viz_fig' in st.session_state:
                    st.plotly_chart(st.session_state['clustering_viz_fig'], use_container_width=True, key="clustering_viz_saved")

                    # Download buttons
                    st.write("**下载可视化**")
                    col_dl1, col_dl2 = st.columns(2)

                    with col_dl1:
                        html_buffer = io.StringIO()
                        st.session_state['clustering_viz_fig'].write_html(html_buffer)
                        html_str = html_buffer.getvalue()
                        st.download_button(
                            "📥 下载HTML",
                            data=html_str,
                            file_name=f"clustering_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True
                        )

                    with col_dl2:
                        try:
                            png_bytes = st.session_state['clustering_viz_fig'].to_image(format="png", width=1200, height=800)
                            st.download_button(
                                "📥 下载PNG",
                                data=png_bytes,
                                file_name=f"clustering_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        except:
                            st.caption("⚠️ PNG导出需要安装kaleido包")

            else:
                st.info("👈 请先在左侧执行聚类分析")

        # Export section
        if 'clustering_features_df_clustered' in st.session_state:
            st.markdown("---")
            st.subheader("💾 导出结果")

            clustering_features_df_clustered = st.session_state['clustering_features_df_clustered']

            col_export1, col_export2 = st.columns(2)

            with col_export1:
                csv_data = clustering_features_df_clustered.to_csv(index=False)
                st.download_button(
                    "📥 下载带聚类标签的CSV",
                    data=csv_data,
                    file_name=f"clustering_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="下载包含聚类标签的完整特征数据"
                )

            with col_export2:
                st.info(f"📊 聚类数据: {len(clustering_features_df_clustered)} 个细胞，{len(clustering_features_df_clustered.columns)} 个特征列")


    else:
        st.info("👆 请上传细胞特征CSV文件开始分析")

# ==================== 标签页5: 细胞形态学提取 ====================
with tab5:
    st.header("🔬 细胞形态学特征提取")

    st.markdown("""
    ### 功能说明

    本模块提供两种形态学特征提取方式：

    1. **📊 CSV数据处理**：上传已有的特征CSV文件，进行统计分析和可视化
    2. **🔍 直接特征提取**：上传图像和分割掩码，直接提取形态学特征

    ---
    """)

    # 模式选择
    morphology_mode = st.radio(
        "选择处理模式",
        ["📊 CSV数据处理", "🔍 直接特征提取"],
        horizontal=True,
        help="选择特征提取的方式"
    )

    # ==================== 模式1: CSV数据处理 ====================
    if morphology_mode == "📊 CSV数据处理":
        st.subheader("📊 CSV数据处理")

        st.info("""
        **使用说明**：
        - 上传之前导出的细胞特征CSV文件
        - 系统将自动分析并显示统计信息
        - 支持特征分布可视化
        - 可以重新导出处理后的数据
        """)

        # CSV文件上传
        uploaded_csv = st.file_uploader(
            "上传细胞特征CSV文件",
            type=["csv"],
            key="morphology_csv_upload",
            help="上传之前导出的细胞特征CSV文件"
        )

        if uploaded_csv is not None:
            try:
                # 读取CSV文件
                features_df = pd.read_csv(uploaded_csv)

                st.success(f"✅ 成功加载CSV文件，包含 {len(features_df)} 个细胞的特征数据")

                # 显示数据预览
                with st.expander("📋 数据预览", expanded=False):
                    st.dataframe(features_df.head(10), use_container_width=True)
                    st.caption(f"显示前10行数据，共 {len(features_df)} 行")

                # 特征统计分析
                st.subheader("📊 特征统计摘要")

                # 检查是否包含基础特征列
                basic_features = ['area_um2', 'circularity', 'major_axis_length', 'minor_axis_length']
                has_basic_features = all(col in features_df.columns for col in basic_features)

                if has_basic_features:
                    # 计算统计信息
                    stats = get_feature_statistics(features_df)

                    # 显示关键特征的统计信息
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("平均面积", f"{stats['area_um2']['mean']:.1f} μm²")
                    with col2:
                        st.metric("平均圆度", f"{stats['circularity']['mean']:.3f}")
                    with col3:
                        st.metric("平均长轴", f"{stats['major_axis_length']['mean']:.1f} μm")
                    with col4:
                        st.metric("平均短轴", f"{stats['minor_axis_length']['mean']:.1f} μm")

                    # 显示详细统计表格
                    with st.expander("📈 详细统计信息", expanded=False):
                        stats_df = pd.DataFrame(stats).T
                        st.dataframe(stats_df.round(3), use_container_width=True)
                else:
                    st.warning("⚠️ CSV文件缺少基础特征列，无法显示统计摘要")

                # 特征分布可视化
                st.subheader("📊 特征分布可视化")

                if has_basic_features:
                    # 选择要可视化的特征
                    viz_feature = st.selectbox(
                        "选择要可视化的特征",
                        options=['area_um2', 'circularity', 'major_axis_length', 'minor_axis_length',
                                'perimeter_um', 'eccentricity', 'solidity', 'aspect_ratio'],
                        format_func=lambda x: {
                            'area_um2': '面积 (μm²)',
                            'circularity': '圆度',
                            'major_axis_length': '长轴长度 (μm)',
                            'minor_axis_length': '短轴长度 (μm)',
                            'perimeter_um': '周长 (μm)',
                            'eccentricity': '离心率',
                            'solidity': '实心度',
                            'aspect_ratio': '长宽比'
                        }.get(x, x)
                    )

                    if viz_feature in features_df.columns:
                        # 创建直方图
                        fig = px.histogram(
                            features_df,
                            x=viz_feature,
                            nbins=30,
                            title=f"{viz_feature} 分布",
                            labels={viz_feature: viz_feature}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"⚠️ 特征 '{viz_feature}' 不存在于CSV文件中")

                # CSV下载功能
                st.subheader("💾 导出数据")

                col_download1, col_download2 = st.columns(2)

                with col_download1:
                    # 下载原始CSV
                    csv_data = features_df.to_csv(index=False)
                    st.download_button(
                        "📥 下载CSV文件",
                        data=csv_data,
                        file_name=f"morphology_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="下载完整的特征数据CSV文件"
                    )

                with col_download2:
                    if has_basic_features:
                        # 下载统计摘要
                        stats_csv = pd.DataFrame(stats).T.to_csv()
                        st.download_button(
                            "📊 下载统计摘要",
                            data=stats_csv,
                            file_name=f"morphology_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help="下载特征统计摘要CSV文件"
                        )

            except Exception as e:
                st.error(f"❌ 读取CSV文件时出错: {str(e)}")
                st.info("请确保上传的是有效的CSV文件，并且包含正确的特征列")

        else:
            st.info("👆 请上传细胞特征CSV文件开始分析")

    # ==================== 模式2: 直接特征提取 ====================
    elif morphology_mode == "🔍 直接特征提取":
        st.subheader("🔍 直接特征提取")

        st.info("""
        **使用说明**：
        - 上传原始图像和分割掩码
        - 选择特征提取方式（基础/高级）
        - 系统将自动提取细胞形态学特征
        - 支持导出CSV文件
        """)

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.write("**⚙️ 设置**")

            # 图像上传
            uploaded_image = st.file_uploader(
                "上传原始图像",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="morphology_image_upload",
                help="上传细胞的原始图像"
            )

            # 掩码上传
            uploaded_mask = st.file_uploader(
                "上传分割掩码",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="morphology_mask_upload",
                help="上传分割后的掩码图像（标记图像）"
            )

            # 像素大小设置
            st.write("**📏 像素大小设置**")
            pixel_size_morph = st.number_input(
                "像素大小 (μm/pixel)",
                min_value=0.01,
                max_value=10.0,
                value=0.65,
                step=0.01,
                key="morphology_pixel_size",
                help="输入显微镜的像素大小，用于计算实际物理尺寸"
            )

            # 最小细胞面积设置
            min_area_morph = st.number_input(
                "最小细胞面积 (像素)",
                min_value=10,
                max_value=10000,
                value=100,
                step=10,
                key="morphology_min_area",
                help="过滤掉小于此面积的区域"
            )

            # 特征提取选项
            st.write("**🔬 特征提取选项**")
            use_advanced_morph = st.checkbox(
                "使用高级特征提取",
                value=False,
                key="morphology_advanced",
                help="提取更多高级特征（需要原始图像）"
            )

            # 高级特征选项
            if use_advanced_morph:
                with st.expander("🔬 高级特征选项", expanded=True):
                    include_hu_morph = st.checkbox("Hu矩特征", value=True, key="morphology_hu")
                    include_intensity_morph = st.checkbox("强度特征", value=True, key="morphology_intensity")
                    include_texture_morph = st.checkbox("纹理特征", value=True, key="morphology_texture")
                    include_boundary_morph = st.checkbox("边界特征", value=True, key="morphology_boundary")
                    include_advanced_shape_morph = st.checkbox("高级形状特征", value=True, key="morphology_advanced_shape")

            # 提取按钮
            extract_button = st.button(
                "🚀 开始提取特征",
                use_container_width=True,
                type="primary",
                disabled=(uploaded_mask is None)
            )

        with col_right:
            st.write("**📊 结果显示**")

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
                        st.warning("⚠️ 掩码中未检测到细胞区域")
                    else:
                        st.info(f"🔍 检测到 {num_cells} 个细胞区域")

                        with st.spinner("正在提取细胞特征..."):
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
                                    st.success("✅ 高级特征提取完成！")
                                else:
                                    st.error("❌ 高级特征提取需要上传原始图像")
                                    features_df = None
                            else:
                                # 使用基础特征提取
                                features_df = extract_cell_features(
                                    labeled_mask,
                                    pixel_size=pixel_size_morph,
                                    min_area=min_area_morph
                                )
                                st.success("✅ 基础特征提取完成！")

                            if features_df is not None and not features_df.empty:
                                # 显示特征统计
                                st.subheader("📊 特征统计摘要")
                                stats = get_feature_statistics(features_df)

                                # 显示关键特征的统计信息
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("平均面积", f"{stats['area_um2']['mean']:.1f} μm²")
                                with col2:
                                    st.metric("平均圆度", f"{stats['circularity']['mean']:.3f}")
                                with col3:
                                    st.metric("平均长轴", f"{stats['major_axis_length']['mean']:.1f} μm")
                                with col4:
                                    st.metric("平均短轴", f"{stats['minor_axis_length']['mean']:.1f} μm")

                                # 显示详细特征表格
                                with st.expander("📋 查看详细特征数据", expanded=False):
                                    if use_advanced_morph:
                                        # 高级特征模式：显示所有列
                                        st.caption("💡 **提示**：表格支持横向滚动，可以拖动查看所有列")
                                        st.dataframe(features_df.round(3), use_container_width=True, height=400)
                                    else:
                                        # 基础特征模式：只显示主要列
                                        display_cols = ['sequential_id', 'cell_id', 'area_um2', 'perimeter_um', 'circularity',
                                                      'major_axis_length', 'minor_axis_length', 'eccentricity',
                                                      'solidity', 'aspect_ratio']
                                        st.dataframe(features_df[display_cols].round(3), use_container_width=True, height=400)

                                # CSV下载功能
                                st.subheader("💾 导出数据")

                                col_export1, col_export2 = st.columns(2)

                                with col_export1:
                                    # 下载特征CSV
                                    csv_data = features_df.to_csv(index=False)
                                    st.download_button(
                                        "📥 下载特征CSV",
                                        data=csv_data,
                                        file_name=f"cell_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                        help="下载完整的细胞特征数据CSV文件"
                                    )

                                with col_export2:
                                    # 下载统计摘要
                                    stats_csv = pd.DataFrame(stats).T.to_csv()
                                    st.download_button(
                                        "📊 下载统计摘要",
                                        data=stats_csv,
                                        file_name=f"feature_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                        help="下载特征统计摘要CSV文件"
                                    )

                            else:
                                st.warning("⚠️ 未能提取到有效的特征数据")

                except Exception as e:
                    st.error(f"❌ 处理图像时出错: {str(e)}")
                    st.info("请确保上传的是有效的图像文件")

            else:
                if uploaded_mask is None:
                    st.info("👆 请上传分割掩码开始特征提取")
                else:
                    st.info("👈 点击左侧的'开始提取特征'按钮")
