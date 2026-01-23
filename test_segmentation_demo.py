"""
传统分割算法演示脚本

演示阈值分割、边缘检测、形态学操作、分水岭分割和轮廓分析功能
"""
import numpy as np
import cv2
from pathlib import Path

from src.data.image_io import ImageIO
from src.core.segmentation.threshold import ThresholdSegmentation
from src.core.segmentation.edge import EdgeDetection
from src.core.segmentation.morphology import MorphologicalOps
from src.core.segmentation.watershed import WatershedSegmentation
from src.core.segmentation.contour import ContourAnalysis
from src.core.utils.logger import setup_logger
from src.core.utils.paths import ensure_dir

# 设置日志
logger = setup_logger(level="INFO")
logger.info("开始传统分割算法演示...")

# 创建测试目录
test_dir = ensure_dir("test_segmentation_output")
logger.info(f"测试输出目录: {test_dir}")

# 1. 创建测试图像
logger.info("\n=== 步骤1: 创建测试图像 ===")
height, width = 256, 256
test_image = np.zeros((height, width, 3), dtype=np.uint8)

# 添加一些几何图形（模拟细胞）
cv2.circle(test_image, (64, 64), 30, (200, 200, 200), -1)
cv2.circle(test_image, (192, 64), 25, (200, 200, 200), -1)
cv2.circle(test_image, (64, 192), 28, (200, 200, 200), -1)
cv2.circle(test_image, (192, 192), 32, (200, 200, 200), -1)
cv2.circle(test_image, (128, 128), 35, (200, 200, 200), -1)

# 添加一些粘连的细胞
cv2.ellipse(test_image, (100, 180), (25, 15), 45, 0, 360, (200, 200, 200), -1)
cv2.ellipse(test_image, (120, 190), (20, 12), -30, 0, 360, (200, 200, 200), -1)

# 添加轻微噪声
noise = np.random.randint(0, 30, test_image.shape, dtype=np.uint8)
test_image = cv2.add(test_image, noise)

logger.info(f"创建测试图像: shape={test_image.shape}")

# 保存原始图像
ImageIO.save_image(test_image, test_dir / "01_original.png")
logger.info("✓ 保存原始图像")

# 转换为灰度图
gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
ImageIO.save_image(gray_image, test_dir / "02_grayscale.png")
logger.info("✓ 保存灰度图像")

# 2. 阈值分割演示
logger.info("\n=== 步骤2: 阈值分割演示 ===")

# Otsu阈值分割
otsu_binary = ThresholdSegmentation.otsu_threshold(gray_image)
ImageIO.save_image(otsu_binary, test_dir / "03_threshold_otsu.png")
logger.info("✓ Otsu阈值分割完成")

# 固定阈值分割
fixed_binary = ThresholdSegmentation.fixed_threshold(gray_image, threshold=100)
ImageIO.save_image(fixed_binary, test_dir / "04_threshold_fixed.png")
logger.info("✓ 固定阈值分割完成")

# 自适应阈值分割
adaptive_binary = ThresholdSegmentation.adaptive_threshold(gray_image, block_size=15)
ImageIO.save_image(adaptive_binary, test_dir / "05_threshold_adaptive.png")
logger.info("✓ 自适应阈值分割完成")

# 3. 边缘检测演示
logger.info("\n=== 步骤3: 边缘检测演示 ===")

# Canny边缘检测
canny_edges = EdgeDetection.canny(gray_image, threshold1=50, threshold2=150)
ImageIO.save_image(canny_edges, test_dir / "06_edge_canny.png")
logger.info("✓ Canny边缘检测完成")

# Sobel边缘检测
sobel_edges = EdgeDetection.sobel(gray_image, dx=1, dy=1, ksize=3)
ImageIO.save_image(sobel_edges, test_dir / "07_edge_sobel.png")
logger.info("✓ Sobel边缘检测完成")

# Laplacian边缘检测
laplacian_edges = EdgeDetection.laplacian(gray_image, ksize=3)
ImageIO.save_image(laplacian_edges, test_dir / "08_edge_laplacian.png")
logger.info("✓ Laplacian边缘检测完成")

# 4. 形态学操作演示
logger.info("\n=== 步骤4: 形态学操作演示 ===")

# 使用Otsu二值图像进行形态学操作
# 腐蚀
eroded = MorphologicalOps.erode(otsu_binary, kernel_size=(5, 5))
ImageIO.save_image(eroded, test_dir / "09_morph_erode.png")
logger.info("✓ 腐蚀操作完成")

# 膨胀
dilated = MorphologicalOps.dilate(otsu_binary, kernel_size=(5, 5))
ImageIO.save_image(dilated, test_dir / "10_morph_dilate.png")
logger.info("✓ 膨胀操作完成")

# 开运算
opened = MorphologicalOps.opening(otsu_binary, kernel_size=(5, 5))
ImageIO.save_image(opened, test_dir / "11_morph_opening.png")
logger.info("✓ 开运算完成")

# 闭运算
closed = MorphologicalOps.closing(otsu_binary, kernel_size=(5, 5))
ImageIO.save_image(closed, test_dir / "12_morph_closing.png")
logger.info("✓ 闭运算完成")

# 5. 分水岭分割演示
logger.info("\n=== 步骤5: 分水岭分割演示 ===")

# 使用距离变换的分水岭分割
watershed_labels = WatershedSegmentation.watershed_distance_transform(
    closed, min_distance=20
)
# 可视化分水岭结果
watershed_vis = WatershedSegmentation.visualize_watershed(
    test_image, watershed_labels, show_boundaries=True
)
ImageIO.save_image(watershed_vis, test_dir / "13_watershed_distance.png")
logger.info("✓ 距离变换分水岭分割完成")

# 标记控制的分水岭分割
watershed_markers = WatershedSegmentation.watershed_marker_controlled(
    gray_image, closed, sure_fg_erosion=5, sure_bg_dilation=5
)
watershed_vis2 = WatershedSegmentation.visualize_watershed(
    test_image, watershed_markers, show_boundaries=True
)
ImageIO.save_image(watershed_vis2, test_dir / "14_watershed_marker.png")
logger.info("✓ 标记控制分水岭分割完成")

# 6. 轮廓检测和分析演示
logger.info("\n=== 步骤6: 轮廓检测和分析演示 ===")

# 查找轮廓
contours = ContourAnalysis.find_contours(closed, mode='external')
logger.info(f"找到 {len(contours)} 个轮廓")

# 过滤小轮廓
filtered_contours = ContourAnalysis.filter_contours(contours, min_area=200)
logger.info(f"过滤后剩余 {len(filtered_contours)} 个轮廓")

# 绘制轮廓
contour_image = ContourAnalysis.draw_contours(
    test_image, filtered_contours, color=(0, 255, 0), thickness=2
)
ImageIO.save_image(contour_image, test_dir / "15_contours.png")
logger.info("✓ 轮廓绘制完成")

# 分析轮廓属性
if len(filtered_contours) > 0:
    logger.info("\n轮廓属性分析:")
    for i, contour in enumerate(filtered_contours[:5]):  # 只显示前5个
        props = ContourAnalysis.get_contour_properties(contour)
        logger.info(f"  轮廓 {i+1}:")
        logger.info(f"    - 面积: {props['area']:.2f}")
        logger.info(f"    - 周长: {props['perimeter']:.2f}")
        logger.info(f"    - 圆形度: {props['circularity']:.3f}")
        logger.info(f"    - 长宽比: {props['aspect_ratio']:.3f}")

# 获取边界框
boxes = ContourAnalysis.get_bounding_boxes(filtered_contours)
bbox_image = test_image.copy()
for box in boxes:
    x, y, w, h = box
    cv2.rectangle(bbox_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
ImageIO.save_image(bbox_image, test_dir / "16_bounding_boxes.png")
logger.info("✓ 边界框绘制完成")

# 总结
logger.info("\n" + "="*50)
logger.info("✅ 所有传统分割算法演示完成！")
logger.info(f"测试结果已保存到: {test_dir.absolute()}")
logger.info("="*50)
