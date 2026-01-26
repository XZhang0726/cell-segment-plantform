"""
测试Dempster-Shafer理论融合功能

这个脚本测试DST融合的基本功能，包括：
1. DST模块导入
2. 质量函数计算
3. Dempster组合规则
4. 融合结果验证
"""

import numpy as np
from src.core.fusion import DempsterShaferFusion, FusionResult

def test_dst_import():
    """测试1: 验证DST模块可以正确导入"""
    print("=" * 60)
    print("测试1: DST模块导入")
    print("=" * 60)

    try:
        from src.core.fusion import (
            DempsterShaferFusion,
            FusionResult,
            handle_conflict,
            generate_conflict_map
        )
        print("[PASS] DST模块导入成功")
        return True
    except Exception as e:
        print(f"[FAIL] DST模块导入失败: {e}")
        return False

def test_mass_function():
    """测试2: 验证质量函数计算"""
    print("\n" + "=" * 60)
    print("测试2: 质量函数计算")
    print("=" * 60)

    try:
        # 初始化DST引擎
        fusion_engine = DempsterShaferFusion({
            'cellpose': 0.9,
            'cellvit': 0.85,
            'cellsam': 0.8
        })

        # 测试质量函数计算
        mass = fusion_engine.compute_mass_from_confidence('cellpose', 0.8)

        print(f"模型: cellpose, 置信度: 0.8, 可靠性: 0.9")
        print(f"质量函数: {mass}")

        # 验证质量函数和为1
        total = sum(mass.values())
        print(f"质量函数总和: {total:.6f}")

        if abs(total - 1.0) < 1e-6:
            print("[PASS] 质量函数计算正确")
            return True
        else:
            print(f"[FAIL] 质量函数总和不为1: {total}")
            return False

    except Exception as e:
        print(f"[FAIL] 质量函数计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dempster_combination():
    """测试3: 验证Dempster组合规则"""
    print("\n" + "=" * 60)
    print("测试3: Dempster组合规则")
    print("=" * 60)

    try:
        fusion_engine = DempsterShaferFusion({
            'model1': 0.9,
            'model2': 0.85
        })

        # 两个质量函数
        m1 = {'Cell': 0.7, 'Background': 0.2, 'Cell|Background': 0.1}
        m2 = {'Cell': 0.6, 'Background': 0.3, 'Cell|Background': 0.1}

        print(f"质量函数1: {m1}")
        print(f"质量函数2: {m2}")

        # 组合
        combined, conflict = fusion_engine.dempster_combine(m1, m2)

        print(f"\n组合结果: {combined}")
        print(f"冲突系数: {conflict:.3f}")

        # 验证组合结果和为1
        total = sum(combined.values())
        print(f"组合质量函数总和: {total:.6f}")

        if abs(total - 1.0) < 1e-6 and 0 <= conflict < 1:
            print("[PASS] Dempster组合规则正确")
            return True
        else:
            print(f"[FAIL] 组合结果异常")
            return False

    except Exception as e:
        print(f"[FAIL] Dempster组合失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_instance_fusion():
    """测试4: 验证实例融合"""
    print("\n" + "=" * 60)
    print("测试4: 实例融合")
    print("=" * 60)

    try:
        fusion_engine = DempsterShaferFusion({
            'cellpose': 0.9,
            'cellvit': 0.85,
            'cellsam': 0.8
        })

        # 模拟3个模型的预测
        matched_group = [
            ('cellpose', 0.8),
            ('cellvit', 0.6),
            ('cellsam', 0.9)
        ]

        print(f"匹配组: {matched_group}")

        # 执行融合
        result = fusion_engine.fuse_instances(matched_group)

        print(f"\n融合结果:")
        print(f"  决策: {result.decision}")
        print(f"  置信度: {result.confidence:.3f}")
        print(f"  冲突度: {result.conflict:.3f}")
        print(f"  不确定性: {result.uncertainty:.3f}")
        print(f"  信念区间: [{result.belief_cell:.3f}, {result.plausibility_cell:.3f}]")

        # 验证结果合理性
        if (result.decision in ['Cell', 'Background', 'Cell|Background'] and
            0 <= result.confidence <= 1 and
            0 <= result.conflict < 1 and
            0 <= result.uncertainty <= 1):
            print("[PASS] 实例融合成功")
            return True
        else:
            print("[FAIL] 融合结果异常")
            return False

    except Exception as e:
        print(f"[FAIL] 实例融合失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conflict_scenarios():
    """测试5: 验证冲突场景处理"""
    print("\n" + "=" * 60)
    print("测试5: 冲突场景处理")
    print("=" * 60)

    try:
        fusion_engine = DempsterShaferFusion({
            'model1': 0.9,
            'model2': 0.9
        })

        # 场景1: 高一致性（低冲突）
        print("\n场景1: 高一致性")
        group1 = [('model1', 0.9), ('model2', 0.85)]
        result1 = fusion_engine.fuse_instances(group1)
        print(f"  冲突度: {result1.conflict:.3f} (预期: <0.3)")

        # 场景2: 中等冲突
        print("\n场景2: 中等冲突")
        group2 = [('model1', 0.8), ('model2', 0.3)]
        result2 = fusion_engine.fuse_instances(group2)
        print(f"  冲突度: {result2.conflict:.3f} (预期: 0.3-0.6)")

        # 场景3: 高冲突
        print("\n场景3: 高冲突")
        group3 = [('model1', 0.9), ('model2', 0.1)]
        result3 = fusion_engine.fuse_instances(group3)
        print(f"  冲突度: {result3.conflict:.3f} (预期: >0.6)")

        if result1.conflict < result2.conflict < result3.conflict:
            print("\n[PASS] 冲突场景处理正确")
            return True
        else:
            print("\n[FAIL] 冲突度排序异常")
            return False

    except Exception as e:
        print(f"[FAIL] 冲突场景测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Dempster-Shafer理论融合功能测试")
    print("=" * 60)

    results = []

    # 运行测试
    results.append(("DST模块导入", test_dst_import()))
    results.append(("质量函数计算", test_mass_function()))
    results.append(("Dempster组合规则", test_dempster_combination()))
    results.append(("实例融合", test_instance_fusion()))
    results.append(("冲突场景处理", test_conflict_scenarios()))

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
        print("\n[SUCCESS] 所有测试通过！DST融合功能正常工作。")
        return True
    else:
        print(f"\n[WARN] {total - passed} 个测试失败，请检查实现。")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
