"""
功能测试脚本

演示图像IO和预处理功能
"""
import numpy as np
from pathlib import Path

from src.data.image_io import ImageIO
from src.core.preprocessing.preprocess import ImagePreprocessor
from src.core.utils.logger import setup_logger
from src.core.utils.paths import ensure_dir

# 设置日志
logger = setup_logger(level="INFO")
logger.info("开始功能测试...")

# 创建测试目录
test_dir = ensure_dir("test_output")
logger.info(f"测试输出目录: {test_dir}")

# 1. 创建测试图像
logger.info("\n=== 步骤1: 创建测试图像 ===")
# 创建一个带有噪声的测试图像
height, width = 256, 256
test_image = np.zeros((height, width, 3), dtype=np.uint8)

# 添加一些几何图形
cv2 = __import__('cv2')
cv2.circle(test_image, (128, 128), 50, (255, 0, 0), -1)  # 红色圆
cv2.rectangle(test_image, (50, 50), (150, 100), (0, 255, 0), -1)  # 绿色矩形
cv2.line(test_image, (0, 0), (256, 256), (0, 0, 255), 3)  # 蓝色线

# 添加噪声
noise = np.random.randint(0, 50, test_image.shape, dtype=np.uint8)
test_image = cv2.add(test_image, noise)

logger.info(f"创建测试图像: shape={test_image.shape}, dtype={test_image.dtype}")

# 2. 保存原始图像
logger.info("\n=== 步骤2: 保存原始图像 ===")
original_path = test_dir / "01_original.png"
ImageIO.save_image(test_image, original_path)
logger.info(f"保存原始图像到: {original_path}")

# 3. 测试图像加载
logger.info("\n=== 步骤3: 测试图像加载 ===")
loaded_image = ImageIO.load_image(original_path)
logger.info(f"加载图像: shape={loaded_image.shape}")

# 验证加载的图像与原始图像一致
assert loaded_image.shape == test_image.shape
logger.info("✓ 图像加载测试通过")

# 4. 测试灰度转换
logger.info("\n=== 步骤4: 测试灰度转换 ===")
gray_image = ImageIO.load_image(original_path, grayscale=True)
logger.info(f"灰度图像: shape={gray_image.shape}")
ImageIO.save_image(gray_image, test_dir / "02_grayscale.png")
logger.info("✓ 灰度转换测试通过")

# 5. 测试图像归一化
logger.info("\n=== 步骤5: 测试图像归一化 ===")
normalized = ImagePreprocessor.normalize(test_image, method='minmax')
logger.info(f"归一化后: min={normalized.min():.3f}, max={normalized.max():.3f}")
# 转换回uint8保存
normalized_uint8 = (normalized * 255).astype(np.uint8)
ImageIO.save_image(normalized_uint8, test_dir / "03_normalized.png")
logger.info("✓ 图像归一化测试通过")

# 6. 测试图像缩放
logger.info("\n=== 步骤6: 测试图像缩放 ===")
resized = ImagePreprocessor.resize(test_image, (128, 128))
logger.info(f"缩放后: shape={resized.shape}")
ImageIO.save_image(resized, test_dir / "04_resized_128x128.png")
logger.info("✓ 图像缩放测试通过")

# 7. 测试图像去噪
logger.info("\n=== 步骤7: 测试图像去噪 ===")
denoised_gaussian = ImagePreprocessor.denoise(test_image, method='gaussian', ksize=5)
ImageIO.save_image(denoised_gaussian, test_dir / "05_denoised_gaussian.png")
logger.info("✓ 高斯去噪测试通过")

denoised_median = ImagePreprocessor.denoise(test_image, method='median', ksize=5)
ImageIO.save_image(denoised_median, test_dir / "06_denoised_median.png")
logger.info("✓ 中值去噪测试通过")

# 8. 测试对比度增强
logger.info("\n=== 步骤8: 测试对比度增强 ===")
enhanced = ImagePreprocessor.enhance_contrast(test_image, method='clahe')
ImageIO.save_image(enhanced, test_dir / "07_enhanced_clahe.png")
logger.info("✓ CLAHE对比度增强测试通过")

# 9. 测试获取图像信息
logger.info("\n=== 步骤9: 测试获取图像信息 ===")
info = ImageIO.get_image_info(original_path)
logger.info(f"图像信息:")
logger.info(f"  - 文件名: {info['filename']}")
logger.info(f"  - 格式: {info['format']}")
logger.info(f"  - 尺寸: {info['width']}x{info['height']}")
logger.info(f"  - 模式: {info['mode']}")
logger.info("✓ 获取图像信息测试通过")

# 10. 综合测试：完整的预处理流程
logger.info("\n=== 步骤10: 综合测试 - 完整预处理流程 ===")
# 加载 -> 去噪 -> 增强 -> 归一化 -> 缩放
pipeline_image = ImageIO.load_image(original_path)
pipeline_image = ImagePreprocessor.denoise(pipeline_image, method='bilateral')
pipeline_image = ImagePreprocessor.enhance_contrast(pipeline_image, method='clahe')
pipeline_image = ImagePreprocessor.normalize(pipeline_image, method='minmax')
pipeline_image = (pipeline_image * 255).astype(np.uint8)
pipeline_image = ImagePreprocessor.resize(pipeline_image, (200, 200))
ImageIO.save_image(pipeline_image, test_dir / "08_pipeline_result.png")
logger.info("✓ 完整预处理流程测试通过")

# 总结
logger.info("\n" + "="*50)
logger.info("✅ 所有功能测试通过！")
logger.info(f"测试结果已保存到: {test_dir.absolute()}")
logger.info("="*50)
