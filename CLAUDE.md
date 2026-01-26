# ⚠️ 重要：环境配置说明

## 🚨 强制要求

**本项目必须在 `cellpose_gpu` 虚拟环境下运行！**

**绝对不要在 base 环境下运行！**

---

## ✅ RTX 5070 GPU 完全支持 (已解决)

**好消息：RTX 5070 GPU 现已完全支持！**

- ✅ **解决方案**：PyTorch 2.10.0+cu128 已支持 RTX 5070 (sm_120 计算能力)
- ✅ **CUDA版本**：需要 CUDA 12.8 支持
- ✅ **状态**：GPU加速已完全启用，性能显著提升
- ✅ **验证**：已通过完整的GPU功能测试

**当前状态：RTX 5070 GPU 已完全配置并可用于 Cellpose 加速处理。**

---

## 为什么必须使用 cellpose_gpu 环境？

1. **Python版本兼容性**
   - base环境使用Python 3.13
   - PyTorch GPU版本需要Python 3.12
   - cellpose_gpu环境使用Python 3.12

2. **GPU加速支持**
   - cellpose_gpu环境安装了PyTorch GPU版本
   - 支持CUDA 12.1，可以使用RTX 5070 GPU
   - Cellpose深度学习模型需要GPU加速（否则极慢）

3. **依赖隔离**
   - 避免污染base环境
   - 避免包版本冲突
   - 便于管理和维护

---

## 📋 环境创建步骤（首次使用）

如果还没有创建 `cellpose_gpu` 环境，请按以下步骤操作：

```bash
# 1. 创建环境（Python 3.12）
mamba create -n cellpose_gpu python=3.12 -y

# 2. 激活环境
conda activate cellpose_gpu

# 3. 安装PyTorch GPU版本 (CUDA 12.8 for RTX 5070 support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. 安装项目依赖
pip install cellpose segment-anything streamlit loguru opencv-python scikit-image pandas openpyxl

# 5. 验证GPU可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 6. 运行完整GPU测试（推荐）
python test_gpu_cellpose.py
```

---

## 🚀 每次运行项目前的步骤

### 1. 激活环境

```bash
conda activate cellpose_gpu
```

### 2. 进入项目目录

```bash
cd c:\Users\XB001\Desktop\cc_works\xibaofenge
```

### 3. 运行项目

```bash
# 运行Streamlit应用
streamlit run app_enhanced.py

# 或运行性能测试
python test_parallel_performance.py
```

---

## ✅ 检查当前环境

运行以下命令检查是否在正确的环境中：

```bash
# 查看当前环境
conda env list

# 查看Python版本（应该是3.12）
python --version

# 查看PyTorch和CUDA状态
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

**正确的输出应该是：**
- Python版本：3.12.x
- PyTorch版本：2.10.0+cu128 或更高版本
- CUDA available: True
- GPU名称：NVIDIA GeForce RTX 5070

---

## ❌ 常见错误

### 错误1：在base环境运行

```
❌ 错误：(base) C:\Users\XB001>streamlit run app_enhanced.py
```

**解决方法：**
```bash
conda activate cellpose_gpu
```

### 错误2：CUDA不可用

```
❌ 错误：CUDA available: False
```

**原因：**
- 可能在base环境（Python 3.13）
- 或者安装了CPU版本的PyTorch

**解决方法：**
1. 确认在cellpose_gpu环境
2. 重新安装PyTorch GPU版本

### 错误3：Cellpose运行极慢

**原因：**
- GPU未启用
- 在CPU模式下运行

**解决方法：**
1. 检查CUDA是否可用
2. 确认在cellpose_gpu环境
3. 在应用中启用GPU选项

---

## 📝 快速参考

| 操作 | 命令 |
|------|------|
| 激活环境 | `conda activate cellpose_gpu` |
| 退出环境 | `conda deactivate` |
| 查看环境列表 | `conda env list` |
| 查看已安装包 | `conda list` 或 `pip list` |
| 运行应用 | `streamlit run app_enhanced.py` |

---

## 🔧 环境管理

### 删除环境（如需重建）

```bash
conda deactivate
mamba env remove -n cellpose_gpu
```

### 导出环境配置

```bash
conda activate cellpose_gpu
conda env export > environment.yml
```

### 从配置文件创建环境

```bash
mamba env create -f environment.yml
```

---

## 💡 提示

1. **每次打开新终端都要激活环境**
2. **看到 `(cellpose_gpu)` 前缀才是正确的**
3. **GPU加速可以让Cellpose快10-50倍**
4. **遇到问题先检查环境是否正确**

---

## 📞 问题排查

如果遇到问题，按以下顺序检查：

1. ✅ 是否激活了cellpose_gpu环境？
2. ✅ Python版本是否为3.12？
3. ✅ PyTorch是否为GPU版本（+cu121）？
4. ✅ CUDA是否可用（torch.cuda.is_available()）？
5. ✅ 所有依赖是否已安装？

---

**最后提醒：永远不要在base环境运行本项目！**

**Always activate `cellpose_gpu` before running anything!**
