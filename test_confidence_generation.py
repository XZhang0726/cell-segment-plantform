"""
测试置信度生成功能

验证合成置信度图的生成是否正确
"""

import numpy as np
import matplotlib.pyplot as plt
from src.core.fusion.confidence_utils import generate_confidence_from_mask, generate_confidence_maps

def test_single_mask_confidence():
    """测试单个掩码的置信度生成"""
    print("=" * 60)
    print("测试1: 单个掩码的置信度生成")
    print("=" * 60)

    # 创建一个简单的测试掩码（100x100，包含2个圆形细胞）
    mask = np.zeros((100, 100), dtype=np.int32)

    # 细胞1：中心在(30, 30)，半径15
    y, x = np.ogrid[:100, :100]
    cell1 = ((x - 30)**2 + (y - 30)**2) <= 15**2
    mask[cell1] = 1

    # 细胞2：中心在(70, 70)，半径20
    cell2 = ((x - 70)**2 + (y - 70)**2) <= 20**2
    mask[cell2] = 2

    # 生成置信度图
    confidence_map = generate_confidence_from_mask(mask, base_confidence=0.8, boundary_penalty=0.3)

    # 验证结果
    print(f"掩码形状: {mask.shape}")
    print(f"细胞数量: {np.max(mask)}")
    print(f"置信度图形状: {confidence_map.shape}")
    print(f"置信度范围: [{np.min(confidence_map):.3f}, {np.max(confidence_map):.3f}]")

    # 检查细胞1的置信度分布
    cell1_conf = confidence_map[cell1]
    print(f"\n细胞1置信度统计:")
    print(f"  平均: {np.mean(cell1_conf):.3f}")
    print(f"  最小: {np.min(cell1_conf):.3f} (边界)")
    print(f"  最大: {np.max(cell1_conf):.3f} (中心)")

    # 检查细胞2的置信度分布
    cell2_conf = confidence_map[cell2]
    print(f"\n细胞2置信度统计:")
    print(f"  平均: {np.mean(cell2_conf):.3f}")
    print(f"  最小: {np.min(cell2_conf):.3f} (边界)")
    print(f"  最大: {np.max(cell2_conf):.3f} (中心)")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mask, cmap='tab20')
    axes[0].set_title('原始掩码')
    axes[0].axis('off')

    axes[1].imshow(confidence_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('置信度图')
    axes[1].axis('off')

    # 叠加显示
    axes[2].imshow(mask, cmap='gray', alpha=0.3)
    im = axes[2].imshow(confidence_map, cmap='hot', alpha=0.7, vmin=0, vmax=1)
    axes[2].set_title('叠加显示')
    axes[2].axis('off')

    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('test_confidence_single.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: test_confidence_single.png")

    # 验证：中心置信度应该高于边界
    if np.max(cell1_conf) > np.min(cell1_conf):
        print("\n[PASS] 置信度梯度正确：中心 > 边界")
        return True
    else:
        print("\n[FAIL] 置信度梯度异常")
        return False


def test_multiple_models_confidence():
    """测试多个模型的置信度生成"""
    print("\n" + "=" * 60)
    print("测试2: 多个模型的置信度生成")
    print("=" * 60)

    # 创建3个模型的掩码（模拟不同模型的分割结果）
    mask1 = np.zeros((100, 100), dtype=np.int32)
    mask2 = np.zeros((100, 100), dtype=np.int32)
    mask3 = np.zeros((100, 100), dtype=np.int32)

    y, x = np.ogrid[:100, :100]

    # 模型1：检测到2个细胞
    cell1 = ((x - 30)**2 + (y - 30)**2) <= 15**2
    cell2 = ((x - 70)**2 + (y - 70)**2) <= 20**2
    mask1[cell1] = 1
    mask1[cell2] = 2

    # 模型2：检测到2个细胞（位置略有不同）
    cell1_shifted = ((x - 32)**2 + (y - 32)**2) <= 16**2
    cell2_shifted = ((x - 68)**2 + (y - 68)**2) <= 19**2
    mask2[cell1_shifted] = 1
    mask2[cell2_shifted] = 2

    # 模型3：只检测到1个细胞
    mask3[cell2] = 1

    masks_list = [mask1, mask2, mask3]
    model_names = ['cellpose', 'cellvit', 'cellsam']
    model_reliabilities = {
        'cellpose': 0.9,
        'cellvit': 0.85,
        'cellsam': 0.8
    }

    # 生成置信度图
    confidences_list = generate_confidence_maps(masks_list, model_names, model_reliabilities)

    print(f"生成了 {len(confidences_list)} 个置信度图")

    # 验证每个模型的置信度
    for i, (conf_map, model_name) in enumerate(zip(confidences_list, model_names)):
        mask = masks_list[i]
        reliability = model_reliabilities[model_name]

        # 计算前景区域的平均置信度
        foreground = mask > 0
        if np.sum(foreground) > 0:
            avg_conf = np.mean(conf_map[foreground])
            print(f"\n{model_name}:")
            print(f"  可靠性: {reliability:.2f}")
            print(f"  平均置信度: {avg_conf:.3f}")
            print(f"  置信度范围: [{np.min(conf_map[foreground]):.3f}, {np.max(conf_map[foreground]):.3f}]")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, (mask, conf_map, model_name) in enumerate(zip(masks_list, confidences_list, model_names)):
        # 第一行：掩码
        axes[0, i].imshow(mask, cmap='tab20')
        axes[0, i].set_title(f'{model_name} - 掩码')
        axes[0, i].axis('off')

        # 第二行：置信度图
        im = axes[1, i].imshow(conf_map, cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title(f'{model_name} - 置信度')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('test_confidence_multiple.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: test_confidence_multiple.png")

    # 验证：不同模型应该有不同的置信度分布
    conf1_mean = np.mean(confidences_list[0][masks_list[0] > 0])
    conf2_mean = np.mean(confidences_list[1][masks_list[1] > 0])
    conf3_mean = np.mean(confidences_list[2][masks_list[2] > 0])

    if conf1_mean != conf2_mean and conf2_mean != conf3_mean:
        print("\n[PASS] 不同模型的置信度分布不同")
        return True
    else:
        print("\n[FAIL] 置信度分布异常")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("置信度生成功能测试")
    print("=" * 60)

    results = []

    # 运行测试
    results.append(("单个掩码置信度生成", test_single_mask_confidence()))
    results.append(("多个模型置信度生成", test_multiple_models_confidence()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS] 通过" if result else "[FAIL] 失败"
        print(f"{test_name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n[SUCCESS] 所有测试通过！置信度生成功能正常工作。")
        return True
    else:
        print(f"\n[WARN] {total - passed} 个测试失败，请检查实现。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
