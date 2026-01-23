"""
日志配置模块

使用loguru提供统一的日志管理功能
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_file: str = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_string: str = None
):
    """
    配置日志系统

    Args:
        log_file: 日志文件路径，如果为None则只输出到控制台
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: 日志轮转大小
        retention: 日志保留时间
        format_string: 自定义日志格式
    """
    # 移除默认的handler
    logger.remove()

    # 默认格式
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # 添加控制台输出
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True
    )

    # 如果指定了日志文件，添加文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )

    return logger


def get_logger(name: str = None):
    """
    获取logger实例

    Args:
        name: logger名称

    Returns:
        logger实例
    """
    if name:
        return logger.bind(name=name)
    return logger
