"""
细胞分割平台 - Gradio UI界面

提供易用的Web界面进行细胞图像分割
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import gradio as gr
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.segmentation import CellSegmenter, SegmentationMethod


def segment_cell_image(
    image,
    method,
    block_size=11,
    C=2,
    low_threshold=50,
    high_threshold=150,
    model_path=None
):
    """
    分割细胞图像

    Args:
        image: 输入图像
        method: 分割方法
        block_size: 自适应阈值的块大小
        C: 自适应阈值的常数
        low_threshold: Canny边缘检测的低阈值
        high_threshold: Canny边缘检测的高阈值
        model_path: 深度学习模型路径

    Returns:
        原图和分割结果
    """
    if image is None:
        return None, None, "请上传图像"

    # 转换图像格式
    if isinstance(image, Image.Image):
        image = np.array(image)

    # 确保图像是uint8类型
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    try:
        # 创建分割器
        method_map = {
            "Otsu阈值": SegmentationMethod.OTSU,
            "自适应阈值": SegmentationMethod.ADAPTIVE,
            "分水岭算法": SegmentationMethod.WATERSHED,
            "Canny边缘检测": SegmentationMethod.EDGE_CANNY,
            "深度学习": SegmentationMethod.DEEP_LEARNING
        }

        seg_method = method_map[method]
        segmenter = CellSegmenter(
            method=seg_method,
            model_path=model_path if seg_method == SegmentationMethod.DEEP_LEARNING else None
        )

        # 根据方法设置参数
        params = {}
        if seg_method == SegmentationMethod.ADAPTIVE:
            params = {"block_size": int(block_size), "C": int(C)}
        elif seg_method == SegmentationMethod.EDGE_CANNY:
            params = {"low_threshold": int(low_threshold), "high_threshold": int(high_threshold)}

        # 执行分割
        mask = segmenter.segment(image, **params)

        # 归一化掩码用于显示
        if mask.max() > 0:
            mask_display = (mask / mask.max() * 255).astype(np.uint8)
        else:
            mask_display = mask.astype(np.uint8)

        # 创建彩色叠加图
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()

        # 创建红色掩码叠加
        overlay = image_rgb.copy()
        overlay[mask > 0] = [255, 0, 0]  # 红色
        result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

        # 统计信息
        foreground_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        foreground_ratio = foreground_pixels / total_pixels * 100

        info = f"""
        ### 分割结果统计
        - **方法**: {method}
        - **前景像素数**: {foreground_pixels:,}
        - **总像素数**: {total_pixels:,}
        - **前景比例**: {foreground_ratio:.2f}%
        """

        if seg_method == SegmentationMethod.WATERSHED:
            num_regions = len(np.unique(mask)) - 1
            info += f"\n- **检测到的区域数**: {num_regions}"

        return mask_display, result, info

    except Exception as e:
        return None, None, f"错误: {str(e)}"


# 创建Gradio界面
with gr.Blocks(title="细胞分割平台") as demo:
    gr.Markdown("""
    # 🔬 细胞分割平台

    上传细胞显微镜图像，选择分割方法，实时查看分割结果
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # 输入区域
            gr.Markdown("### 📤 输入图像")
            input_image = gr.Image(
                label="上传细胞图像",
                type="numpy",
                height=300
            )

            gr.Markdown("### ⚙️ 分割设置")
            method = gr.Dropdown(
                choices=["Otsu阈值", "自适应阈值", "分水岭算法", "Canny边缘检测"],
                value="Otsu阈值",
                label="分割方法"
            )

            # 参数设置（根据方法显示不同参数）
            with gr.Accordion("高级参数", open=False):
                block_size = gr.Slider(
                    minimum=3,
                    maximum=51,
                    step=2,
                    value=11,
                    label="自适应阈值 - 块大小 (block_size)",
                    info="必须是奇数，越大越平滑"
                )
                C = gr.Slider(
                    minimum=0,
                    maximum=20,
                    step=1,
                    value=2,
                    label="自适应阈值 - 常数 (C)",
                    info="从平均值中减去的常数"
                )
                low_threshold = gr.Slider(
                    minimum=0,
                    maximum=200,
                    step=10,
                    value=50,
                    label="Canny - 低阈值",
                    info="边缘检测的低阈值"
                )
                high_threshold = gr.Slider(
                    minimum=0,
                    maximum=300,
                    step=10,
                    value=150,
                    label="Canny - 高阈值",
                    info="边缘检测的高阈值"
                )

            segment_btn = gr.Button("🚀 开始分割", variant="primary", size="lg")

        with gr.Column(scale=2):
            # 输出区域
            gr.Markdown("### 📊 分割结果")

            with gr.Row():
                mask_output = gr.Image(label="分割掩码", height=300)
                overlay_output = gr.Image(label="叠加显示", height=300)

            info_output = gr.Markdown()

    # 示例图像
    gr.Markdown("### 💡 示例")
    gr.Examples(
        examples=[
            ["Otsu阈值", 11, 2, 50, 150],
            ["自适应阈值", 15, 3, 50, 150],
            ["分水岭算法", 11, 2, 50, 150],
            ["Canny边缘检测", 11, 2, 30, 100],
        ],
        inputs=[method, block_size, C, low_threshold, high_threshold],
        label="预设参数组合"
    )

    # 使用说明
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown("""
        ### 使用步骤
        1. **上传图像**: 点击上传区域选择细胞显微镜图像
        2. **选择方法**: 从下拉菜单选择分割方法
        3. **调整参数**: 展开"高级参数"调整方法特定的参数
        4. **开始分割**: 点击"开始分割"按钮
        5. **查看结果**: 查看分割掩码和叠加显示

        ### 方法说明
        - **Otsu阈值**: 自动选择最佳阈值，适合对比度高的图像
        - **自适应阈值**: 局部自适应阈值，适合光照不均的图像
        - **分水岭算法**: 基于距离变换的区域分割，可分离粘连细胞
        - **Canny边缘检测**: 检测细胞边缘，适合边界清晰的图像

        ### 参数建议
        - **block_size**: 建议11-35之间的奇数，图像越大可以用越大的值
        - **C**: 建议2-10之间，值越大检测越保守
        - **Canny阈值**: low建议20-50，high建议60-150
        """)

    # 绑定事件
    segment_btn.click(
        fn=segment_cell_image,
        inputs=[input_image, method, block_size, C, low_threshold, high_threshold],
        outputs=[mask_output, overlay_output, info_output]
    )


if __name__ == "__main__":
    # 使用默认设置启动
    demo.launch(
        share=False,
        inbrowser=False
    )
