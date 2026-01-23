"""
路径处理工具模块

提供项目中常用的路径操作功能
"""
from pathlib import Path
from typing import Union, List


def get_project_root() -> Path:
    """
    获取项目根目录

    Returns:
        项目根目录路径
    """
    # 从当前文件向上查找，直到找到包含setup.py的目录
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists():
            return parent
    # 如果没找到，返回当前文件的上上上级目录
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    """获取数据目录"""
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """获取模型目录"""
    return get_project_root() / "models"


def get_configs_dir() -> Path:
    """获取配置目录"""
    return get_project_root() / "configs"


def get_results_dir() -> Path:
    """获取结果目录"""
    return get_data_dir() / "results"


def list_files(
    directory: Union[str, Path],
    extensions: List[str] = None,
    recursive: bool = False
) -> List[Path]:
    """
    列出目录中的文件

    Args:
        directory: 目录路径
        extensions: 文件扩展名列表，如['.png', '.jpg']
        recursive: 是否递归搜索子目录

    Returns:
        文件路径列表
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    files = []
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if extensions is None or file_path.suffix.lower() in extensions:
                files.append(file_path)

    return sorted(files)


def get_relative_path(path: Union[str, Path], base: Union[str, Path] = None) -> Path:
    """
    获取相对路径

    Args:
        path: 目标路径
        base: 基准路径，默认为项目根目录

    Returns:
        相对路径
    """
    path = Path(path).resolve()
    if base is None:
        base = get_project_root()
    else:
        base = Path(base).resolve()

    try:
        return path.relative_to(base)
    except ValueError:
        # 如果路径不在base下，返回绝对路径
        return path
