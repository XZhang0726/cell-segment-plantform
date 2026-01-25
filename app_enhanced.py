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

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.segmentation import CellSegmenter, SegmentationMethod
from src.core.features import extract_cell_features, get_feature_statistics, extract_advanced_cell_features

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
    individual_cells = []
    cell_info = []

    # 获取所有唯一标签（排除背景0）
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]

    for label in unique_labels:
        # 创建当前细胞的二值掩码
        cell_mask_binary = (mask == label).astype(np.uint8)

        # 计算面积
        area = np.sum(cell_mask_binary)

        # 过滤太小的区域
        if area < min_area:
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
        "Cellpose深度学习": SegmentationMethod.CELLPOSE
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
            # 对于标签掩码（Cellpose、分水岭等），保留区域标签
            binary_mask = (mask > 0).astype(np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            mask = mask * binary_mask
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

    # 归一化掩码
    if mask.max() > 0:
        mask_display = (mask / mask.max() * 255).astype(np.uint8)
    else:
        mask_display = mask.astype(np.uint8)

    # 创建叠加图
    if len(processed_image.shape) == 2:
        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = processed_image.copy()

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

        #### 1. 图像分割（支持单图和对比模式）
        - **单图模式**: 选择一种分割方法处理图像，可调整方法参数
        - **对比模式**: 启用"对比模式"同时使用多种方法，并排显示结果
        - **分割方法**:
          - 传统方法: Otsu阈值、自适应阈值、分水岭算法、Canny边缘检测
          - 深度学习: Cellpose（推荐用于复杂细胞图像和重叠细胞分割）
        - **预处理选项**: 去噪、对比度增强、归一化
        - **结果导出**: 下载分割掩码和叠加图

        #### 2. 批量处理
        - 一次上传多张图像进行批量分割
        - 自动应用相同的分割参数和预处理选项
        - 实时显示处理进度
        - 可视化查看每张图像的分割结果
        - 批量导出掩码、叠加图和统计报告（ZIP格式）

        #### 3. 预处理选项
        - **去噪处理**: 使用高斯滤波减少图像噪声
        - **对比度增强**: 使用CLAHE算法增强局部对比度
        - **归一化**: 将像素值归一化到标准范围

        #### 4. 后处理选项
        - **区域闭合**: 使用形态学闭运算填充细胞边界间隙，获得完整的细胞区域
        - **提取单个细胞**: 自动提取每个细胞样本，支持机器学习训练数据准备
        - **最小细胞面积**: 过滤掉面积过小的噪声区域

        ### 使用技巧
        - **Cellpose深度学习**: 对于重叠或接触的细胞，推荐使用Cellpose方法，效果最佳
        - **模型选择**: cyto2适合大多数细胞质染色图像，nuclei适合细胞核染色图像
        - **预处理**: 对于噪声较大的图像，建议启用去噪和对比度增强
        - **区域闭合**: 默认启用，可填补细胞边界的小间隙，获得更完整的分割结果
        - **细胞提取**: 启用后可导出单个细胞样本，适合用于深度学习模型训练
        - **批量处理**: 适合处理大量相似类型的细胞图像
        - **对比模式**: 不确定哪种方法最适合时，启用对比模式快速评估
        - **参数调整**: 根据图像特点调整方法参数以获得最佳效果
        """)

# 创建标签页
tab1, tab2 = st.tabs(["📤 图像分割", "📦 批量处理"])

# ==================== 标签页1: 图像分割（整合单图和对比） ====================
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

        # 对比模式开关
        comparison_mode = st.checkbox("🔍 启用对比模式", value=False, help="同时使用多种方法进行分割对比")

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

        if not comparison_mode:
            # 单一方法模式
            st.subheader("📐 分割方法")
            method = st.selectbox(
                "选择方法",
                ["Otsu阈值", "自适应阈值", "分水岭算法", "Canny边缘检测", "Cellpose深度学习"],
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
                        tile_norm_blocksize = st.slider("归一化块大小", 0, 256, 0, 16,
                                                       help="0表示全局归一化，>0表示分块归一化")
                        normalize = {"tile_norm_blocksize": tile_norm_blocksize}
                    else:
                        normalize = None

                params = {"model_type": model_type, "diameter": diameter, "use_gpu": use_gpu,
                         "batch_size": batch_size, "normalize": normalize}
            else:
                params = {}
        else:
            # 对比模式
            st.subheader("📊 对比方法选择")
            st.write("选择要对比的方法（至少2个）：")
            comp_methods = []
            if st.checkbox("Otsu阈值", value=True, key="comp_otsu_tab1"):
                comp_methods.append("Otsu阈值")
            if st.checkbox("自适应阈值", value=True, key="comp_adaptive_tab1"):
                comp_methods.append("自适应阈值")
            if st.checkbox("分水岭算法", value=False, key="comp_watershed_tab1"):
                comp_methods.append("分水岭算法")
            if st.checkbox("Canny边缘检测", value=False, key="comp_canny_tab1"):
                comp_methods.append("Canny边缘检测")
            if st.checkbox("Cellpose深度学习", value=False, key="comp_cellpose_tab1"):
                comp_methods.append("Cellpose深度学习")

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

                        if not comparison_mode:
                            # 单一方法模式
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

                        else:
                            # 对比模式
                            if len(comp_methods) < 2:
                                st.warning("⚠️ 请至少选择2种方法进行对比")
                            else:
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
                        st.error(f"❌ 分割失败: {str(e)}")
        else:
            st.info("👈 请在左侧上传细胞图像")

# ==================== 标签页2: 批量处理 ====================
with tab2:
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
            ["Otsu阈值", "自适应阈值", "分水岭算法", "Canny边缘检测", "Cellpose深度学习"],
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

