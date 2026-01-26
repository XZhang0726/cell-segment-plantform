"""
实例匹配性能对比测试

对比原始版本和优化版本的性能差异
"""
import time
import numpy as np
from pathlib import Path
import cv2
from loguru import logger

# 导入原始版本和优化版本
from src.core.fusion.instance_matcher import match_instances as match_instances_original
from src.core.fusion.instance_matcher_optimized import match_instances_optimized


def load_test_image(image_path: str) -> np.ndarray:
    """加载测试图片（支持中文路径）"""
    # 使用 numpy.fromfile 读取文件，避免中文路径问题
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"无法加载图片: {image_path}")

    # 转换为灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    logger.info(f"加载图片: {image_path}, 尺寸: {img.shape}")
    return img


def simulate_segmentation_masks(image: np.ndarray, num_models: int = 3) -> list:
    """
    模拟多个模型的分割结果

    为了测试性能，我们创建模拟的分割mask
    每个模型会生成略有不同的细胞检测结果
    """
    height, width = image.shape
    masks = []

    # 使用简单的阈值分割作为基础
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 距离变换和分水岭
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    for model_idx in range(num_models):
        # 每个模型使用稍微不同的阈值
        threshold_ratio = 0.3 + model_idx * 0.1
        _, markers = cv2.threshold(dist_transform, threshold_ratio * dist_transform.max(), 255, 0)
        markers = markers.astype(np.uint8)

        # 连通组件标记
        num_labels, labels = cv2.connectedComponents(markers)

        # 应用分水岭
        labels = labels.astype(np.int32)
        cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), labels)

        # 清理标签（移除边界标记-1）
        labels[labels == -1] = 0

        masks.append(labels)
        logger.info(f"模型{model_idx}: 检测到 {num_labels-1} 个实例")

    return masks


def run_performance_test(masks_list: list, iou_threshold: float = 0.5):
    """
    运行性能对比测试

    Args:
        masks_list: 多个模型的分割mask列表
        iou_threshold: IoU匹配阈值
    """
    logger.info("=" * 80)
    logger.info("开始性能对比测试")
    logger.info("=" * 80)

    # 测试原始版本
    logger.info("\n【测试原始版本】")
    start_time = time.time()
    try:
        results_original = match_instances_original(masks_list, iou_threshold)
        time_original = time.time() - start_time
        logger.info(f"✓ 原始版本完成，耗时: {time_original:.4f}秒")
        logger.info(f"  匹配到 {len(results_original)} 个实例组")
    except Exception as e:
        logger.error(f"✗ 原始版本执行失败: {e}")
        time_original = None
        results_original = None

    # 测试优化版本
    logger.info("\n【测试优化版本】")
    start_time = time.time()
    try:
        results_optimized = match_instances_optimized(masks_list, iou_threshold)
        time_optimized = time.time() - start_time
        logger.info(f"✓ 优化版本完成，耗时: {time_optimized:.4f}秒")
        logger.info(f"  匹配到 {len(results_optimized)} 个实例组")
    except Exception as e:
        logger.error(f"✗ 优化版本执行失败: {e}")
        time_optimized = None
        results_optimized = None

    # 性能对比
    logger.info("\n" + "=" * 80)
    logger.info("【性能对比结果】")
    logger.info("=" * 80)

    if time_original and time_optimized:
        speedup = time_original / time_optimized
        logger.info(f"原始版本耗时: {time_original:.4f}秒")
        logger.info(f"优化版本耗时: {time_optimized:.4f}秒")
        logger.info(f"加速比: {speedup:.2f}x")
        logger.info(f"性能提升: {(speedup-1)*100:.1f}%")

        if speedup > 10:
            logger.info("🚀 优化效果显著！")
        elif speedup > 5:
            logger.info("✓ 优化效果良好")
        else:
            logger.info("⚠ 优化效果一般")

    # 结果一致性检查
    if results_original and results_optimized:
        logger.info("\n【结果一致性检查】")
        if len(results_original) == len(results_optimized):
            logger.info(f"✓ 匹配组数量一致: {len(results_original)}")
        else:
            logger.warning(f"⚠ 匹配组数量不一致: 原始={len(results_original)}, 优化={len(results_optimized)}")

    logger.info("=" * 80)


def main():
    """主函数"""
    # 查找测试图片
    test_image_dir = Path(r"C:\Users\XB001\Desktop\细胞")

    if not test_image_dir.exists():
        logger.error(f"测试图片目录不存在: {test_image_dir}")
        return

    # 查找第一张图片
    image_files = list(test_image_dir.glob("*.jpg")) + \
                  list(test_image_dir.glob("*.png")) + \
                  list(test_image_dir.glob("*.tif"))

    if not image_files:
        logger.error(f"在 {test_image_dir} 中未找到图片文件")
        return

    test_image_path = str(image_files[0])
    logger.info(f"使用测试图片: {test_image_path}")

    # 加载图片
    image = load_test_image(test_image_path)

    # 模拟多个模型的分割结果
    logger.info("\n生成模拟分割结果...")
    masks_list = simulate_segmentation_masks(image, num_models=3)

    # 运行性能测试
    run_performance_test(masks_list, iou_threshold=0.5)


if __name__ == "__main__":
    main()
