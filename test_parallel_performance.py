"""
快速性能测试脚本 - 对比串行和并行处理速度
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import numpy as np
import time
from PIL import Image

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.segmentation import CellSegmenter, SegmentationMethod

def create_test_images(num_images=5, size=(512, 512)):
    """创建测试图像"""
    print(f"创建 {num_images} 张测试图像...")
    images = []
    for i in range(num_images):
        # 创建随机噪声图像
        img = np.random.randint(0, 255, size, dtype=np.uint8)
        # 添加一些圆形"细胞"
        for _ in range(20):
            center_x = np.random.randint(50, size[0]-50)
            center_y = np.random.randint(50, size[1]-50)
            radius = np.random.randint(10, 30)
            y, x = np.ogrid[:size[0], :size[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            img[mask] = 200
        images.append(img)
    return images

def test_sequential(images, method=SegmentationMethod.OTSU):
    """测试串行处理"""
    print(f"\n{'='*50}")
    print(f"串行处理测试 - {len(images)} 张图像")
    print(f"{'='*50}")

    segmenter = CellSegmenter(method=method)
    start_time = time.time()

    results = []
    for i, img in enumerate(images):
        print(f"处理图像 {i+1}/{len(images)}...", end='\r')
        mask = segmenter.segment(img)
        results.append(mask)

    elapsed = time.time() - start_time
    print(f"\n串行处理完成: {elapsed:.2f} 秒")
    print(f"平均每张: {elapsed/len(images):.2f} 秒")

    return elapsed, results

def test_parallel(images, method=SegmentationMethod.OTSU):
    """测试并行处理（模拟批量处理）"""
    print(f"\n{'='*50}")
    print(f"并行处理测试 - {len(images)} 张图像")
    print(f"{'='*50}")

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    max_workers = max(1, cpu_count // 2)
    print(f"使用 {max_workers} 个进程（总CPU核心数: {cpu_count}）")

    def process_image(img):
        segmenter = CellSegmenter(method=method)
        return segmenter.segment(img)

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image, images))

    elapsed = time.time() - start_time
    print(f"并行处理完成: {elapsed:.2f} 秒")
    print(f"平均每张: {elapsed/len(images):.2f} 秒")

    return elapsed, results

def main():
    print("="*60)
    print("细胞分割并行处理性能测试")
    print("="*60)

    # 创建测试图像
    num_images = 8  # 测试图像数量
    images = create_test_images(num_images)

    # 使用Otsu方法测试（快速）
    method = SegmentationMethod.OTSU
    print(f"\n使用方法: Otsu阈值分割")

    # 串行测试
    time_sequential, _ = test_sequential(images, method)

    # 并行测试
    time_parallel, _ = test_parallel(images, method)

    # 性能对比
    print(f"\n{'='*60}")
    print("性能对比结果")
    print(f"{'='*60}")
    print(f"串行处理: {time_sequential:.2f} 秒")
    print(f"并行处理: {time_parallel:.2f} 秒")

    if time_parallel < time_sequential:
        speedup = time_sequential / time_parallel
        improvement = ((time_sequential - time_parallel) / time_sequential) * 100
        print(f"\n✅ 加速比: {speedup:.2f}x")
        print(f"✅ 性能提升: {improvement:.1f}%")
        print(f"✅ 节省时间: {time_sequential - time_parallel:.2f} 秒")
    else:
        print(f"\n⚠️ 并行处理未显示优势（可能图像太少或处理太快）")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
