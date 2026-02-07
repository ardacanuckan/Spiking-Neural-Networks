"""
Training Callbacks for SNNs
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class Callback(ABC):
    """Base class for training callbacks."""
    
    def on_epoch_start(self, epoch: int, trainer) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer) -> None:
        pass
    
    def on_batch_start(self, batch: int, trainer) -> None:
        pass
    
    def on_batch_end(self, batch: int, metrics: Dict[str, float], trainer) -> None:
        pass
    
    def on_train_start(self, trainer) -> None:
        pass
    
    def on_train_end(self, trainer) -> None:
        pass


class EarlyStopping(Callback):
    """
    Stop training when monitored metric stops improving.
    
    Args:
        monitor: Metric to monitor ('val_loss', 'val_acc', etc.)
        patience: Epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max'
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.stopped_epoch = 0
        self.stop_training = False
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer) -> None:
        current = metrics.get(self.monitor.replace('val_', ''), metrics.get('loss'))
        
        if self.mode == 'min':
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta
        
        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
                self.stopped_epoch = epoch
                print(f"\nEarly stopping triggered at epoch {epoch}")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Args:
        filepath: Path to save checkpoints
        monitor: Metric to monitor
        save_best_only: Only save when metric improves
        mode: 'min' or 'max'
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min'
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        
        # Create directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer) -> None:
        current = metrics.get(self.monitor.replace('val_', ''), metrics.get('loss'))
        
        if self.save_best_only:
            if self.mode == 'min':
                improved = current < self.best
            else:
                improved = current > self.best
            
            if improved:
                self.best = current
                trainer.save_checkpoint(str(self.filepath))
                print(f"  Saved checkpoint: {self.monitor}={current:.4f}")
        else:
            path = self.filepath.parent / f"{self.filepath.stem}_epoch{epoch}{self.filepath.suffix}"
            trainer.save_checkpoint(str(path))


class LearningRateScheduler(Callback):
    """
    Custom learning rate scheduling callback.
    
    Args:
        schedule_fn: Function (epoch) -> learning_rate
    """
    
    def __init__(self, schedule_fn):
        self.schedule_fn = schedule_fn
    
    def on_epoch_start(self, epoch: int, trainer) -> None:
        new_lr = self.schedule_fn(epoch)
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = new_lr


class SpikeRateLogger(Callback):
    """Log spike rates per layer for analysis."""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.spike_rates: Dict[str, list] = {}
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer) -> None:
        if epoch % self.log_interval == 0:
            for name, module in trainer.model.named_modules():
                if hasattr(module, 'v'):
                    if name not in self.spike_rates:
                        self.spike_rates[name] = []
                    # Approximate spike rate from membrane potential distribution
                    if module.v is not None:
                        rate = (module.v > 0.5).float().mean().item()
                        self.spike_rates[name].append(rate)
