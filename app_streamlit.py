"""
细胞分割平台 - Streamlit UI界面

提供易用的Web界面进行细胞图像分割
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.segmentation import CellSegmenter, SegmentationMethod


# 页面配置
st.set_page_config(
    page_title="细胞分割平台",
    page_icon="🔬",
    layout="wide"
)

# 标题
st.title("🔬 细胞分割平台")
st.markdown("上传细胞显微镜图像，选择分割方法，实时查看分割结果")

# 侧边栏 - 输入和参数
with st.sidebar:
    st.header("⚙️ 设置")

    # 图像上传
    uploaded_file = st.file_uploader(
        "上传细胞图像",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="支持PNG、JPG、TIFF格式"
    )

    # 分割方法选择
    method = st.selectbox(
        "分割方法",
        ["Otsu阈值", "自适应阈值", "分水岭算法", "Canny边缘检测"],
        help="选择细胞分割方法"
    )

    # 高级参数
    with st.expander("🔧 高级参数"):
        if method == "自适应阈值":
            block_size = st.slider(
                "块大小 (block_size)",
                min_value=3,
                max_value=51,
                value=11,
                step=2,
                help="必须是奇数，越大越平滑"
            )
            C = st.slider(
                "常数 (C)",
                min_value=0,
                max_value=20,
                value=2,
                help="从平均值中减去的常数"
            )
        elif method == "Canny边缘检测":
            low_threshold = st.slider(
                "低阈值",
                min_value=0,
                max_value=200,
                value=50,
                step=10,
                help="边缘检测的低阈值"
            )
            high_threshold = st.slider(
                "高阈值",
                min_value=0,
                max_value=300,
                value=150,
                step=10,
                help="边缘检测的高阈值"
            )

    # 分割按钮
    segment_button = st.button("🚀 开始分割", type="primary", use_container_width=True)

# 主区域 - 结果显示
if uploaded_file is not None:
    # 读取图像
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # 显示原始图像
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 原始图像")
        st.image(image, use_container_width=True)
        st.caption(f"尺寸: {image_np.shape}")

    # 执行分割
    if segment_button:
        with col2:
            st.subheader("📊 分割结果")

            with st.spinner("正在分割..."):
                try:
                    # 创建分割器
                    method_map = {
                        "Otsu阈值": SegmentationMethod.OTSU,
                        "自适应阈值": SegmentationMethod.ADAPTIVE,
                        "分水岭算法": SegmentationMethod.WATERSHED,
                        "Canny边缘检测": SegmentationMethod.EDGE_CANNY
                    }

                    seg_method = method_map[method]
                    segmenter = CellSegmenter(method=seg_method)

                    # 设置参数
                    params = {}
                    if seg_method == SegmentationMethod.ADAPTIVE:
                        params = {"block_size": int(block_size), "C": int(C)}
                    elif seg_method == SegmentationMethod.EDGE_CANNY:
                        params = {"low_threshold": int(low_threshold), "high_threshold": int(high_threshold)}

                    # 执行分割
                    start_time = time.time()
                    mask = segmenter.segment(image_np, **params)
                    elapsed_time = time.time() - start_time

                    # 归一化掩码
                    if mask.max() > 0:
                        mask_display = (mask / mask.max() * 255).astype(np.uint8)
                    else:
                        mask_display = mask.astype(np.uint8)

                    # 创建彩色叠加
                    if len(image_np.shape) == 2:
                        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                    else:
                        image_rgb = image_np.copy()

                    overlay = image_rgb.copy()
                    overlay[mask > 0] = [255, 0, 0]
                    result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

                    # 显示结果
                    tab1, tab2 = st.tabs(["分割掩码", "叠加显示"])

                    with tab1:
                        st.image(mask_display, use_container_width=True)

                    with tab2:
                        st.image(result, use_container_width=True)

                    # 统计信息
                    st.success("分割完成！")

                    foreground_pixels = np.sum(mask > 0)
                    total_pixels = mask.size
                    foreground_ratio = foreground_pixels / total_pixels * 100

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("前景像素", f"{foreground_pixels:,}")
                    with col_b:
                        st.metric("前景比例", f"{foreground_ratio:.2f}%")
                    with col_c:
                        st.metric("处理时间", f"{elapsed_time*1000:.2f} ms")

                    if seg_method == SegmentationMethod.WATERSHED:
                        num_regions = len(np.unique(mask)) - 1
                        st.info(f"检测到 {num_regions} 个细胞区域")

                except Exception as e:
                    st.error(f"分割失败: {str(e)}")
else:
    st.info("👈 请在左侧上传细胞图像开始分割")

# 使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 使用步骤
    1. 在左侧上传细胞显微镜图像
    2. 选择分割方法
    3. 调整高级参数（可选）
    4. 点击"开始分割"按钮
    5. 查看分割结果和统计信息

    ### 方法说明
    - **Otsu阈值**: 自动选择最佳阈值，适合对比度高的图像
    - **自适应阈值**: 局部自适应阈值，适合光照不均的图像
    - **分水岭算法**: 基于距离变换的区域分割，可分离粘连细胞
    - **Canny边缘检测**: 检测细胞边缘，适合边界清晰的图像
    """)
