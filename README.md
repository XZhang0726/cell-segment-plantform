# 细胞分割平台 (XiBaoFenGe)

一个全面的、自动化的细胞分割平台，基于深度学习和传统图像处理技术。

## 项目简介

本项目旨在为生物医学研究提供高效、准确的细胞图像分割解决方案，支持多种细胞类型和成像方式。

### 核心特性

- 🎯 **多种分割算法**：支持传统方法和深度学习模型（U-Net、Mask R-CNN等）
- 🚀 **高性能处理**：GPU加速，支持批量处理
- 🖥️ **多种交互方式**：CLI、GUI和API接口
- 📊 **完整分析流程**：从图像预处理到结果统计分析
- 🔧 **易于扩展**：模块化设计，方便添加新算法

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+（可选，用于GPU加速）

### 安装

1. 克隆仓库
```bash
git clone https://github.com/yourusername/xibaofenge.git
cd xibaofenge
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 开发模式安装
```bash
pip install -e .
```

## 基本使用

### 命令行界面（开发中）

```bash
# 单张图像分割
xibaofenge segment --input image.png --output result.png

# 批量处理
xibaofenge batch --input-dir ./images --output-dir ./results

# 模型训练
xibaofenge train --config configs/unet.yaml
```

### Python API（开发中）

```python
from xibaofenge import CellSegmenter

# 初始化分割器
segmenter = CellSegmenter(model='unet')

# 加载图像
image = segmenter.load_image('cell_image.png')

# 执行分割
result = segmenter.segment(image)

# 保存结果
segmenter.save_result(result, 'output.png')
```

## 项目结构

```
xibaofenge/
├── src/                    # 源代码
│   ├── core/              # 核心算法
│   ├── data/              # 数据管理
│   ├── training/          # 模型训练
│   ├── inference/         # 推理服务
│   └── ui/                # 用户界面
├── data/                   # 数据目录
├── models/                 # 模型文件
├── configs/                # 配置文件
├── tests/                  # 测试代码
├── docs/                   # 文档
└── notebooks/              # Jupyter notebooks
```

## 开发状态

当前版本：**v0.1.0-alpha**

- [x] 项目框架搭建
- [ ] 图像预处理模块
- [ ] 传统分割算法
- [ ] 深度学习模型集成
- [ ] CLI工具
- [ ] GUI界面

详细开发计划请查看 [docs/00_计划导航.md](docs/00_计划导航.md)

## 文档

- [项目概述](docs/01_项目概述.md)
- [技术栈与架构](docs/02_技术栈与架构.md)
- [开发路线图](docs/03_开发路线图.md)
- [实施计划](docs/04_实施计划.md)

## 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)（待创建）了解详情。

## 许可证

MIT License（待确认）

## 联系方式

- 项目主页：https://github.com/yourusername/xibaofenge
- 问题反馈：https://github.com/yourusername/xibaofenge/issues
