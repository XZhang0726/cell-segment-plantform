# CellViT 深度学习模型使用说明

## 概述

CellViT 是一个基于 Vision Transformer 的细胞核分割模型，主要用于病理切片图像分析。本项目已集成 CellViT 作为深度学习分割方法之一。

## 环境要求

### 专用环境

CellViT 需要在专门的 Python 环境中运行，因为它的依赖与主项目环境（cellpose_gpu）存在版本冲突：

- **主项目环境**: `cellpose_gpu` (numpy 2.x)
- **CellViT 环境**: `env_cellvit` (numpy 1.26.4)

### 硬件要求

- **GPU**: 推荐使用 CUDA 兼容的 GPU（至少 8GB VRAM）
- **内存**: 至少 16GB RAM
- **Python**: 3.12.x

## 环境安装

CellViT 环境已经在项目根目录下创建：`./env_cellvit`

### 激活环境

```bash
# 在项目根目录下
source activate ./env_cellvit
```

### 验证安装

```bash
python -c "import cellvit; print('CellViT installed successfully')"
```

## 使用方式

### 1. 在代码中使用

```python
from src.api.segmentation import CellSegmenter, SegmentationMethod

# 创建 CellViT 分割器
segmenter = CellSegmenter(method=SegmentationMethod.CELLVIT)

# 执行分割
masks = segmenter.segment(
    image,
    model_type='CellViT-256',  # 模型类型
    use_gpu=True,              # 使用 GPU
    target_size=256,           # 目标图像大小
    progress_bar=None          # 可选的进度条
)
```

### 2. 参数说明

- `model_type`: 模型类型，目前支持 "CellViT-256"
- `use_gpu`: 是否使用 GPU 加速（推荐）
- `target_size`: 目标图像大小，默认 256x256
- `progress_bar`: Streamlit 进度条对象（可选）

## 注意事项

### 1. 环境切换

**重要**: 使用 CellViT 时必须在 `env_cellvit` 环境中运行：

```bash
# 错误：在 cellpose_gpu 环境中运行
conda activate cellpose_gpu
python app_enhanced.py  # CellViT 将无法工作

# 正确：在 cellvit 环境中运行
source activate ./env_cellvit
python app_enhanced.py  # CellViT 可以正常工作
```

### 2. 图像大小适配

CellViT 主要为大型病理切片设计，本项目提供了适配层：

- 小于 256x256 的图像会被 padding
- 大于 256x256 的图像会被 resize
- 推理后会自动调整回原始大小

### 3. 性能考虑

- **首次运行**: 需要下载预训练模型（约 500MB），可能需要几分钟
- **GPU 加速**: 强烈推荐使用 GPU，CPU 推理会非常慢
- **批处理**: 对于多张图像，建议使用批处理模式

## 依赖版本

主要依赖包及版本：

```
cellvit==1.0.9
torch==2.10.0+cu128
torchvision==0.25.0+cu128
numpy==1.26.4
opencv-python-headless==4.7.0.72
ray==2.53.0
```

完整依赖列表请查看环境中的 `pip list` 输出。

## 故障排除

### 问题 1: ImportError: CellViT not available

**原因**: 未在 cellvit 环境中运行

**解决方案**:
```bash
source activate ./env_cellvit
```

### 问题 2: CUDA out of memory

**原因**: GPU 内存不足

**解决方案**:
- 使用 CPU 模式：`use_gpu=False`
- 减小图像大小
- 关闭其他占用 GPU 的程序

### 问题 3: 模型下载失败

**原因**: 网络连接问题

**解决方案**:
- 检查网络连接
- 使用代理或 VPN
- 手动下载模型文件

## 与其他模型的比较

| 特性 | Cellpose | CellViT | U-Net |
|------|----------|---------|-------|
| 设计目标 | 通用细胞分割 | 病理切片细胞核 | 通用分割 |
| 输入大小 | 灵活 | 256x256 (优化) | 256x256 |
| GPU 要求 | 中等 | 较高 | 中等 |
| 速度 | 快 | 中等 | 快 |
| 精度 | 高 | 非常高 | 中等 |

## 更新日志

- **2026-01-26**: 初始集成 CellViT 1.0.9
  - 创建专用环境 `env_cellvit`
  - 实现图像大小适配层
  - 集成到统一分割 API

## 参考资料

- [CellViT GitHub](https://github.com/TIO-IKIM/CellViT)
- [CellViT 文档](https://tio-ikim.github.io/CellViT-Inference/)
- [CellViT PyPI](https://pypi.org/project/cellvit/)
