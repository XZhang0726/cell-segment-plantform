"""
CellViT Worker Script - 在cellvit环境中运行

这个脚本设计为在cellvit环境中通过subprocess调用
接收图像数据，执行CellViT推理，返回结果
"""
import sys
import os
import pickle
import numpy as np
from pathlib import Path

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def run_cellvit_inference(input_file, output_file):
    """
    执行CellViT推理

    Args:
        input_file: 输入pickle文件路径，包含图像和参数
        output_file: 输出pickle文件路径，保存推理结果
    """
    import time
    start_time = time.time()

    try:
        import torch
        import torch.nn.functional as F
        from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
        from cellvit.utils.cache_models import cache_cellvit_256
        from cellvit.inference.postprocessing_numpy import DetectionCellPostProcessor
        import cv2

        # 读取输入数据
        with open(input_file, 'rb') as f:
            data = pickle.load(f)

        image = data['image']
        model_type = data['model_type']
        use_gpu = data['use_gpu']
        target_size = data['target_size']

        # 设置设备
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # 输出设备信息
        print(f"[CellViT Worker] 使用设备: {device}")
        print(f"[CellViT Worker] use_gpu参数: {use_gpu}")
        print(f"[CellViT Worker] CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[CellViT Worker] GPU名称: {torch.cuda.get_device_name(0)}")
        sys.stdout.flush()

        # 加载模型
        print(f"[CellViT Worker] 开始加载模型...")
        sys.stdout.flush()
        model_load_start = time.time()

        model_path = cache_cellvit_256()
        checkpoint = torch.load(model_path, map_location=device)

        model = CellViT256(
            model256_path=model_path,
            num_nuclei_classes=6,
            num_tissue_classes=19,
        )

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()

        model_load_time = time.time() - model_load_start
        print(f"[CellViT Worker] 模型加载完成，耗时: {model_load_time:.2f}秒")

        # 预处理图像
        h, w = image.shape[:2]
        original_shape = (h, w)

        # 计算padding
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)

        # 如果图像大于target_size，进行resize
        if h > target_size or w > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
            pad_h = target_size - h
            pad_w = target_size - w

        # Padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 转换为tensor
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor / 255.0

        image_tensor = F.pad(
            image_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0
        )

        image_tensor = image_tensor.unsqueeze(0).to(device)

        # 执行推理
        print(f"[CellViT Worker] 开始推理...")
        inference_start = time.time()

        with torch.no_grad():
            predictions = model.forward(image_tensor, retrieve_tokens=False)

        inference_time = time.time() - inference_start
        print(f"[CellViT Worker] 推理完成，耗时: {inference_time:.2f}秒")

        # 调试：打印predictions中的所有键
        print(f"[CellViT Worker] Predictions keys: {list(predictions.keys())}")
        sys.stdout.flush()

        # 使用CellViT后处理器将HV map转换为实例分割图
        print(f"[CellViT Worker] 开始后处理，将HV map转换为实例分割图...")
        postprocess_start = time.time()

        # 检查predictions的形状并转换为后处理器需要的格式 (B, H, W, C)
        # 当前格式: (B, C, H, W) -> 需要: (B, H, W, C)
        predictions_reshaped = {}
        for key in ['nuclei_binary_map', 'hv_map', 'nuclei_type_map']:
            if key in predictions:
                tensor = predictions[key]
                print(f"[CellViT Worker] {key} 原始形状: {tensor.shape}")
                # 从 (B, C, H, W) 转换为 (B, H, W, C)
                if tensor.dim() == 4:
                    tensor = tensor.permute(0, 2, 3, 1)
                predictions_reshaped[key] = tensor
                print(f"[CellViT Worker] {key} 转换后形状: {tensor.shape}")

        sys.stdout.flush()

        # 创建一个简单的WSI元数据对象（后处理器需要）
        class SimpleWSI:
            def __init__(self):
                self.metadata = {'pixel_size': 1.0}

        # 创建后处理器
        # num_nuclei_classes = 6 (CellViT-256的默认值)
        post_processor = DetectionCellPostProcessor(
            wsi=SimpleWSI(),
            nr_types=6,
            binary=False
        )

        # 执行后处理
        try:
            instance_maps, cell_dicts = post_processor.post_process_batch(predictions_reshaped)
            print(f"[CellViT Worker] 后处理完成，耗时: {time.time() - postprocess_start:.2f}秒")

            # 提取第一个图像的instance map
            instance_map = instance_maps[0].cpu().numpy() if isinstance(instance_maps, torch.Tensor) else instance_maps[0]

            print(f"[CellViT Worker] Instance map形状: {instance_map.shape}")
            print(f"[CellViT Worker] Instance map数据类型: {instance_map.dtype}")
            print(f"[CellViT Worker] Instance map值范围: {instance_map.min()} - {instance_map.max()}")
            print(f"[CellViT Worker] 唯一细胞数量: {len(np.unique(instance_map)) - 1}")
            sys.stdout.flush()

        except Exception as e:
            print(f"[CellViT Worker] 后处理失败: {str(e)}")
            print(f"[CellViT Worker] 回退到使用binary map")
            # 如果后处理失败，回退到使用binary map
            if 'nuclei_binary_map' in predictions:
                instance_map = predictions['nuclei_binary_map'][0, 1].cpu().numpy()
            else:
                raise ValueError("无法生成instance map")
            sys.stdout.flush()

        # 移除padding
        if pad_bottom > 0:
            instance_map = instance_map[pad_top:-pad_bottom, :]
        else:
            instance_map = instance_map[pad_top:, :]

        if pad_right > 0:
            instance_map = instance_map[:, pad_left:-pad_right]
        else:
            instance_map = instance_map[:, pad_left:]

        # 调整回原始大小
        if instance_map.shape != original_shape:
            instance_map = cv2.resize(
                instance_map.astype(np.float32),
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # 保存结果
        result = {
            'success': True,
            'mask': instance_map.astype(np.int32),
            'error': None
        }

        with open(output_file, 'wb') as f:
            pickle.dump(result, f)

        total_time = time.time() - start_time
        print(f"[CellViT Worker] 总处理时间: {total_time:.2f}秒")
        print(f"[CellViT Worker] 检测到 {len(np.unique(instance_map)) - 1} 个细胞核")

        return 0

    except Exception as e:
        # 保存错误信息（包含完整traceback）
        import traceback
        error_traceback = traceback.format_exc()

        # 打印到stderr确保被捕获
        print(f"[CellViT Worker] ERROR: {str(e)}", file=sys.stderr)
        print(f"[CellViT Worker] TRACEBACK:\n{error_traceback}", file=sys.stderr)
        sys.stderr.flush()

        result = {
            'success': False,
            'mask': None,
            'error': f"{str(e)}\n\nTraceback:\n{error_traceback}"
        }

        with open(output_file, 'wb') as f:
            pickle.dump(result, f)

        return 1


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python cellvit_worker.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    exit_code = run_cellvit_inference(input_file, output_file)
    sys.exit(exit_code)
