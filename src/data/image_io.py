"""
图像IO模块

提供图像的加载、保存和基本操作功能
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
from PIL import Image
import skimage.io as skio

from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class ImageIO:
    """图像IO类"""

    # 支持的图像格式
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

    @staticmethod
    def load_image(
        image_path: Union[str, Path],
        grayscale: bool = False,
        backend: str = 'opencv'
    ) -> np.ndarray:
        """
        加载图像

        Args:
            image_path: 图像文件路径
            grayscale: 是否转换为灰度图
            backend: 使用的后端 ('opencv', 'pillow', 'skimage')

        Returns:
            图像数组 (H, W, C) 或 (H, W)

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的图像格式
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if image_path.suffix.lower() not in ImageIO.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        try:
            if backend == 'opencv':
                # OpenCV默认读取为BGR格式
                if grayscale:
                    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                else:
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    # 转换为RGB格式
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif backend == 'pillow':
                image = Image.open(image_path)
                if grayscale:
                    image = image.convert('L')
                else:
                    image = image.convert('RGB')
                image = np.array(image)

            elif backend == 'skimage':
                image = skio.imread(str(image_path), as_gray=grayscale)
                if not grayscale and image.ndim == 2:
                    # 如果是灰度图但要求彩色，转换为3通道
                    image = np.stack([image] * 3, axis=-1)

            else:
                raise ValueError(f"Unknown backend: {backend}")

            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            logger.debug(f"Loaded image: {image_path}, shape: {image.shape}")
            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    @staticmethod
    def save_image(
        image: np.ndarray,
        output_path: Union[str, Path],
        backend: str = 'opencv'
    ) -> None:
        """
        保存图像

        Args:
            image: 图像数组
            output_path: 输出文件路径
            backend: 使用的后端 ('opencv', 'pillow', 'skimage')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if backend == 'opencv':
                # 如果是RGB格式，转换为BGR
                if image.ndim == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), image)

            elif backend == 'pillow':
                if image.ndim == 2:
                    mode = 'L'
                elif image.shape[2] == 3:
                    mode = 'RGB'
                elif image.shape[2] == 4:
                    mode = 'RGBA'
                else:
                    raise ValueError(f"Unsupported image shape: {image.shape}")

                pil_image = Image.fromarray(image, mode=mode)
                pil_image.save(output_path)

            elif backend == 'skimage':
                skio.imsave(str(output_path), image)

            else:
                raise ValueError(f"Unknown backend: {backend}")

            logger.debug(f"Saved image to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            raise

    @staticmethod
    def get_image_info(image_path: Union[str, Path]) -> dict:
        """
        获取图像信息

        Args:
            image_path: 图像文件路径

        Returns:
            包含图像信息的字典
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # 使用PIL获取基本信息（不加载完整图像）
        with Image.open(image_path) as img:
            info = {
                'path': str(image_path),
                'filename': image_path.name,
                'format': img.format,
                'mode': img.mode,
                'size': img.size,  # (width, height)
                'width': img.width,
                'height': img.height,
            }

        return info

    @staticmethod
    def load_images_batch(
        image_paths: List[Union[str, Path]],
        grayscale: bool = False,
        backend: str = 'opencv'
    ) -> List[np.ndarray]:
        """
        批量加载图像

        Args:
            image_paths: 图像文件路径列表
            grayscale: 是否转换为灰度图
            backend: 使用的后端

        Returns:
            图像数组列表
        """
        images = []
        for path in image_paths:
            try:
                image = ImageIO.load_image(path, grayscale=grayscale, backend=backend)
                images.append(image)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue

        logger.info(f"Loaded {len(images)}/{len(image_paths)} images")
        return images


def load_image(image_path: Union[str, Path], grayscale: bool = False) -> np.ndarray:
    """
    加载图像的便捷函数

    Args:
        image_path: 图像文件路径
        grayscale: 是否转换为灰度图

    Returns:
        图像数组
    """
    return ImageIO.load_image(image_path, grayscale=grayscale)


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    保存图像的便捷函数

    Args:
        image: 图像数组
        output_path: 输出文件路径
    """
    ImageIO.save_image(image, output_path)
