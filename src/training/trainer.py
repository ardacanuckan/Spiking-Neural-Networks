"""
SNN Training Pipeline

Comprehensive trainer for Spiking Neural Networks with:
- Temporal unrolling and state management
- Multiple loss function support (CE, TET, temporal contrastive)
- Logging and checkpointing
- Mixed precision training support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any, Tuple
import time
from pathlib import Path
import json


class SNNTrainer:
    """
    Trainer for Spiking Neural Networks.
    
    Handles the temporal dynamics of SNNs by unrolling computation
    across time steps and managing neuron states.
    
    Args:
        model: The SNN model to train
        criterion: Loss function
        optimizer: Optimizer
        time_steps: Number of time steps for temporal unrolling
        device: Device to train on
        surrogate_function: Surrogate gradient function for backprop
        mixed_precision: Whether to use automatic mixed precision
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        time_steps: int = 4,
        device: str = 'cuda',
        surrogate_function: Optional[Callable] = None,
        mixed_precision: bool = False,
        gradient_clip: Optional[float] = None,
        scheduler: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.time_steps = time_steps
        self.device = device
        self.gradient_clip = gradient_clip
        self.scheduler = scheduler
        
        # Default surrogate function
        if surrogate_function is None:
            from ..learning.surrogate import ATan
            surrogate_function = ATan()
        self.surrogate_function = surrogate_function
        
        # Mixed precision
        self.mixed_precision = mixed_precision
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'spike_rate': []
        }
        
        # Callbacks
        self.callbacks: List = []
    
    def add_callback(self, callback) -> None:
        """Add a training callback."""
        self.callbacks.append(callback)
    
    def _reset_model_states(self) -> None:
        """Reset all neuron states in the model."""
        for module in self.model.modules():
            if hasattr(module, 'reset_state'):
                module.reset_state()
    
    def _forward_through_time(
        self, 
        x: torch.Tensor,
        return_spike_trains: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """
        Forward pass through multiple time steps.
        
        Args:
            x: Input tensor [batch, ...]
            return_spike_trains: Whether to return full spike trains
            
        Returns:
            Tuple of (output, spike_trains, firing_rate)
        """
        self._reset_model_states()
        
        outputs = []
        spike_counts = 0
        total_neurons = 0
        
        for t in range(self.time_steps):
            # For static images, use same input at each time step
            # For temporal data, x should be [time, batch, ...]
            if x.dim() > 4:  # Temporal input
                x_t = x[t]
            else:
                x_t = x
            
            # Forward pass
            out = self.model(x_t, self.surrogate_function)
            outputs.append(out)
            
            # Count spikes for monitoring
            if isinstance(out, torch.Tensor):
                spike_counts += out.sum().item()
                total_neurons += out.numel()
        
        # Aggregate outputs across time (rate coding)
        output = torch.stack(outputs, dim=0).mean(dim=0)
        
        # Calculate firing rate
        firing_rate = spike_counts / (total_neurons + 1e-8)
        
        if return_spike_trains:
            spike_trains = torch.stack(outputs, dim=0)
            return output, spike_trains, firing_rate
        
        return output, None, firing_rate
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        total_firing_rate = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output, spike_trains, firing_rate = self._forward_through_time(
                        data, return_spike_trains=True
                    )
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output, spike_trains, firing_rate = self._forward_through_time(
                    data, return_spike_trains=True
                )
                loss = self.criterion(output, target)
                
                loss.backward()
                
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            total_firing_rate += firing_rate
            num_batches += 1
            self.global_step += 1
        
        # Compute epoch metrics
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': 100.0 * correct / total,
            'firing_rate': total_firing_rate / num_batches
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        total_firing_rate = 0.0
        num_batches = 0
        
        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            output, _, firing_rate = self._forward_through_time(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            total_firing_rate += firing_rate
            num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': 100.0 * correct / total,
            'firing_rate': total_firing_rate / num_batches
        }
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        for epoch in range(epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['spike_rate'].append(train_metrics['firing_rate'])
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, {**train_metrics, **val_metrics}, self)
            
            # Logging
            if verbose:
                elapsed = time.time() - start_time
                msg = f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) - "
                msg += f"loss: {train_metrics['loss']:.4f} - "
                msg += f"acc: {train_metrics['accuracy']:.2f}% - "
                msg += f"spike_rate: {train_metrics['firing_rate']:.3f}"
                
                if val_metrics:
                    msg += f" - val_loss: {val_metrics['loss']:.4f} - "
                    msg += f"val_acc: {val_metrics['accuracy']:.2f}%"
                
                print(msg)
        
        return self.history
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
