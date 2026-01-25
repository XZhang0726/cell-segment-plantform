"""
训练器模块

提供用于训练细胞分割模型的训练器类
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import time

from ..core.models.unet import UNet
from .losses import DiceLoss, BCEDiceLoss, FocalLoss
from .metrics import calculate_metrics
from .config import TrainingConfig
from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    训练器类

    负责模型训练、验证和检查点管理
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig
    ):
        """
        初始化训练器

        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 设置设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 初始化损失函数
        self.criterion = self._get_loss_function()

        # 初始化优化器
        self.optimizer = self._get_optimizer()

        # 初始化学习率调度器
        self.scheduler = self._get_scheduler()

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0

        # 创建保存目录
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _get_loss_function(self) -> nn.Module:
        """获取损失函数"""
        loss_type = self.config.loss_type.lower()

        if loss_type == "dice":
            return DiceLoss()
        elif loss_type == "bce_dice":
            return BCEDiceLoss(
                bce_weight=self.config.bce_weight,
                dice_weight=self.config.dice_weight
            )
        elif loss_type == "focal":
            return FocalLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """获取优化器"""
        optimizer_name = self.config.optimizer.lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """获取学习率调度器"""
        scheduler_name = self.config.scheduler.lower()

        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor
            )
        elif scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor
            )
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}, using no scheduler")
            return None

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            包含训练指标的字典
        """
        self.model.train()
        total_loss = 0.0
        total_metrics = {'dice': 0.0, 'iou': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            # 移动数据到设备
            images = images.to(self.device)
            masks = masks.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 记录损失
            total_loss += loss.item()

            # 计算指标
            with torch.no_grad():
                batch_metrics = calculate_metrics(outputs, masks)
                for key in total_metrics:
                    total_metrics[key] += batch_metrics[key]

            # 日志输出
            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Epoch [{self.current_epoch}/{self.config.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Dice: {batch_metrics['dice']:.4f}"
                )

        # 计算平均值
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        avg_metrics['loss'] = avg_loss

        return avg_metrics

    def validate(self) -> Dict[str, float]:
        """
        验证模型

        Returns:
            包含验证指标的字典
        """
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'dice': 0.0, 'iou': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}

        with torch.no_grad():
            for images, masks in self.val_loader:
                # 移动数据到设备
                images = images.to(self.device)
                masks = masks.to(self.device)

                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # 记录损失
                total_loss += loss.item()

                # 计算指标
                batch_metrics = calculate_metrics(outputs, masks)
                for key in total_metrics:
                    total_metrics[key] += batch_metrics[key]

        # 计算平均值
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        avg_metrics['loss'] = avg_loss

        return avg_metrics

    def save_checkpoint(self, filename: str = "checkpoint.pth"):
        """
        保存检查点

        Args:
            filename: 检查点文件名
        """
        checkpoint_path = self.save_dir / filename
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_dice = checkpoint['best_val_dice']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch})")

    def train(self):
        """
        执行完整的训练流程
        """
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 计算epoch时间
            epoch_time = time.time() - start_time

            # 输出epoch总结
            logger.info(
                f"\nEpoch [{self.current_epoch}/{self.config.epochs}] Summary:\n"
                f"  Train Loss: {train_metrics['loss']:.4f} | Train Dice: {train_metrics['dice']:.4f}\n"
                f"  Val Loss: {val_metrics['loss']:.4f} | Val Dice: {val_metrics['dice']:.4f}\n"
                f"  Val IoU: {val_metrics['iou']:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f}\n"
                f"  Learning Rate: {current_lr:.6f} | Time: {epoch_time:.2f}s"
            )

            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint("best_loss.pth")
                logger.info(f"  Saved best loss model (loss: {self.best_val_loss:.4f})")

            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                self.save_checkpoint("best_dice.pth")
                logger.info(f"  Saved best dice model (dice: {self.best_val_dice:.4f})")

            # 定期保存检查点
            if (self.current_epoch) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch}.pth")

        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation dice: {self.best_val_dice:.4f}")

