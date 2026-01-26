# CellSAM 深度学习模型使用说明

## 概述

CellSAM 是基于 Meta 的 Segment Anything Model (SAM) 的细胞分割模型，适用于多种类型的细胞图像。本项目已集成 CellSAM 作为深度学习分割方法之一。

**重要更新**: CellSAM 现在直接在 cellpose_gpu 环境中运行，无需创建专用环境！

## 环境要求

### 运行环境

CellSAM 直接在 **cellpose_gpu** 环境中运行，该环境已包含所有必需的依赖：

- segment-anything 1.0
- PyTorch 2.10.0+cu128
- numpy 2.1.3
- opencv-python
- scipy

### 硬件要求

- **GPU**: 推荐使用 CUDA 兼容的 GPU（至少 4GB VRAM）
- **内存**: 至少 16GB RAM
- **Python**: 3.10+ (cellpose_gpu 环境)

## 安装步骤

### 1. 确认环境

CellSAM 使用 cellpose_gpu 环境，无需额外安装：

```bash
# 激活 cellpose_gpu 环境
source activate cellpose_gpu

# 验证依赖已安装
python -c "import segment_anything; import torch; print('✓ CellSAM dependencies ready')"
```

### 2. 下载模型文件

CellSAM 需要预训练的 SAM 模型检查点文件。请下载并放置在 `models/sam/` 目录下：

```bash
# 创建模型目录
mkdir -p models/sam

# 下载模型（选择一个或多个）
# vit_b (91MB) - 基础模型，推荐首次使用
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/sam/

# vit_l (308MB) - 大模型，更高精度
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P models/sam/

# vit_h (636MB) - 超大模型，最高精度
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/sam/
```

**注意**: 首次使用建议下载 vit_b 模型进行测试。

## 使用方式

### 1. 在代码中使用

```python
from src.api.segmentation import CellSegmenter, SegmentationMethod

# 创建 CellSAM 分割器
segmenter = CellSegmenter(method=SegmentationMethod.CELLSAM)

# 执行分割
masks = segmenter.segment(
    image,
    model_type='vit_b',        # 模型类型: vit_b, vit_l, vit_h
    use_gpu=True,              # 使用 GPU
    points_per_side=32,        # 提示点密度
    progress_bar=None          # 可选的进度条
)
```

### 2. 在 Streamlit 应用中使用

直接在应用界面中选择"CellSAM深度学习"方法即可，无需额外配置。

### 3. 参数说明

- `model_type`: 模型类型
  - `vit_b`: 基础模型 (91M 参数) - 速度快，精度中等
  - `vit_l`: 大模型 (308M 参数) - 平衡速度和精度
  - `vit_h`: 超大模型 (636M 参数) - 精度最高，速度较慢
- `use_gpu`: 是否使用 GPU 加速（推荐）
- `points_per_side`: 自动生成提示点的密度（每边的点数），默认 32
- `progress_bar`: Streamlit 进度条对象（可选）

## 注意事项

### 1. 模型缓存

CellSAM 使用模型缓存机制，首次加载后会保存在内存中，后续使用会更快。

### 2. 模型选择建议

| 模型 | 参数量 | 速度 | 精度 | 推荐场景 |
|------|--------|------|------|----------|
| vit_b | 91M | 快 | 中等 | 快速预览、大批量处理 |
| vit_l | 308M | 中等 | 高 | 平衡速度和精度 |
| vit_h | 636M | 慢 | 最高 | 高精度要求的场景 |

### 3. 性能优化

- **GPU 加速**: 强烈推荐使用 GPU，CPU 推理会非常慢
- **提示点密度**: `points_per_side` 越大，检测到的细胞越多，但处理时间越长
- **批处理**: 对于多张图像，使用批处理模式可以提高效率

## 故障排除

### 问题 1: 模型文件未找到

**现象**: `FileNotFoundError: Model checkpoint not found`

**解决方案**:
```bash
# 下载对应的模型文件到 models/sam/ 目录
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/sam/
```

### 问题 2: CUDA out of memory

**原因**: GPU 内存不足

**解决方案**:
- 使用 CPU 模式：`use_gpu=False`
- 使用更小的模型：`vit_b` 代替 `vit_h`
- 减小提示点密度：`points_per_side=16`
- 关闭其他占用 GPU 的程序

### 问题 3: 检测到的细胞太少

**原因**: 提示点密度不够

**解决方案**:
- 增加提示点密度：`points_per_side=48` 或 `64`
- 使用更大的模型：`vit_l` 或 `vit_h`

### 问题 4: 检测到太多碎片

**原因**: 提示点密度过高或图像噪声

**解决方案**:
- 减小提示点密度：`points_per_side=16` 或 `24`
- 在预处理中启用去噪选项

## 与其他模型的比较

| 特性 | Cellpose | CellViT | CellSAM |
|------|----------|---------|---------|
| 设计目标 | 通用细胞分割 | 病理切片细胞核 | 通用对象分割 |
| 输入大小 | 灵活 | 256x256 (优化) | 灵活 |
| GPU 要求 | 中等 | 较高 (8GB+) | 中等 (4GB+) |
| 速度 | 快 | 中等 | 中等 |
| 精度 | 高 | 非常高 | 高 |
| 适用场景 | 细胞图像 | 病理切片 | 多种图像类型 |
| 环境要求 | cellpose_gpu | env_cellvit (专用) | cellpose_gpu |

## 更新日志

- **2026-01-27**: 简化实现，直接在 cellpose_gpu 环境中运行
  - 移除专用环境 `env_cellsam` 的要求
  - 实现直接导入模式，提升性能
  - 添加模型缓存机制
  - 简化安装和使用流程

## 参考资料

- [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)
- [SAM 论文](https://arxiv.org/abs/2304.02643)
- [SAM 模型下载](https://github.com/facebookresearch/segment-anything#model-checkpoints)

