"""
APEX-SNN: All-Performance EXtreme Spiking Neural Network
=========================================================

The ultimate SNN combining all techniques for maximum performance.

Features:
- TTFS-inspired coding
- Progressive sparsity enforcement
- Multi-scale readouts (ensemble)
- CutMix/Mixup augmentation support
- Learnable tau and thresholds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .surrogate import spike_fn
from .neurons import ParametricLIF, TTFS_LIF
from .nexus_snn import ChebyshevKAN


class SparsityLayer(nn.Module):
    """
    Layer that progressively enforces sparsity during training.
    """
    
    def __init__(self, features, initial_sparsity=0.3, target_sparsity=0.9):
        super().__init__()
        
        self.features = features
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        
        # Learnable importance scores
        self.importance = nn.Parameter(torch.ones(features))
        
        self.register_buffer('current_epoch', torch.tensor(0))
        self.register_buffer('total_epochs', torch.tensor(30))
    
    def set_epoch(self, epoch, total_epochs):
        self.current_epoch = torch.tensor(epoch, device=self.importance.device)
        self.total_epochs = torch.tensor(total_epochs, device=self.importance.device)
    
    def forward(self, x):
        progress = self.current_epoch.float() / (self.total_epochs.float() + 1)
        current_sparsity = self.initial_sparsity + progress * (self.target_sparsity - self.initial_sparsity)
        
        importance_normalized = torch.sigmoid(self.importance)
        
        if self.training:
            mask = importance_normalized
        else:
            threshold = torch.quantile(importance_normalized, current_sparsity)
            mask = (importance_normalized >= threshold).float()
        
        return x * mask.unsqueeze(0)


class APEXLayer(nn.Module):
    """
    Single APEX layer with all enhancements.
    """
    
    def __init__(self, in_features, out_features, tau=2.0, dropout=0.15):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.kan = ChebyshevKAN(out_features, degree=3)
        self.dropout = nn.Dropout(dropout)
        self.lif = ParametricLIF(out_features, tau_init=tau)
        self.sparsity = SparsityLayer(out_features, 0.2, 0.7)
    
    def reset_state(self):
        self.lif.reset_state()
    
    def set_epoch(self, epoch, total_epochs):
        self.sparsity.set_epoch(epoch, total_epochs)
    
    def forward(self, x):
        h = self.bn(self.linear(x))
        h = self.kan(h)
        h = self.sparsity(h)
        h = self.dropout(h)
        spike = self.lif(h)
        return spike, self.lif.v_mem


class APEXSNN(nn.Module):
    """
    APEX-SNN: All-Performance EXtreme Spiking Neural Network.
    
    Architecture:
        Input -> [APEX Layers] -> Multi-scale Readouts -> Ensemble Output
    
    Args:
        input_size: Input dimension
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        tau_range: Range of time constants
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_size=784,
        hidden_sizes=[800, 400, 200],
        num_classes=10,
        tau_range=(1.5, 4.0),
        dropout=0.15
    ):
        super().__init__()
        
        self.hidden_sizes = hidden_sizes
        
        # Time constants
        taus = np.linspace(tau_range[0], tau_range[1], len(hidden_sizes) + 1)
        
        # Input layer
        self.input_linear = nn.Linear(input_size, hidden_sizes[0])
        self.input_bn = nn.BatchNorm1d(hidden_sizes[0])
        self.input_lif = ParametricLIF(hidden_sizes[0], tau_init=taus[0])
        
        # APEX layers
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(
                APEXLayer(hidden_sizes[i], hidden_sizes[i+1], tau=taus[i+1], dropout=dropout)
            )
        
        # Residual projections
        self.residual_projs = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            if hidden_sizes[i] != hidden_sizes[i+1]:
                self.residual_projs.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            else:
                self.residual_projs.append(nn.Identity())
        
        # Skip connection
        self.skip_proj = nn.Linear(hidden_sizes[0], hidden_sizes[-1])
        self.skip_weight = nn.Parameter(torch.tensor(0.2))
        
        # Multi-scale readouts (ensemble)
        self.readouts = nn.ModuleList()
        for i, size in enumerate(hidden_sizes):
            self.readouts.append(nn.Linear(size, num_classes))
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(len(hidden_sizes)) / len(hidden_sizes))
        
        # Output temperature
        self.output_temp = nn.Parameter(torch.tensor(1.0))
    
    def reset_state(self):
        self.input_lif.reset_state()
        for layer in self.layers:
            layer.reset_state()
    
    def set_epoch(self, epoch, total_epochs):
        for layer in self.layers:
            layer.set_epoch(epoch, total_epochs)
    
    def forward(self, x, time_step=0):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Input
        h = self.input_bn(self.input_linear(x))
        spike = self.input_lif(h)
        first_spike = spike
        
        all_spikes = [spike]
        layer_outputs = [spike]
        
        # APEX layers with residuals
        for i, layer in enumerate(self.layers):
            h_new, v_mem = layer(spike)
            residual = self.residual_projs[i](spike)
            spike = h_new + 0.3 * residual
            spike = torch.clamp(spike, 0, 1)
            
            all_spikes.append(spike)
            layer_outputs.append(spike)
        
        # Skip connection
        skip = self.skip_proj(first_spike)
        spike = spike + self.skip_weight.abs() * skip
        spike = torch.clamp(spike, 0, 1)
        
        # Multi-scale readouts
        readout_outputs = []
        for i, (out, readout) in enumerate(zip(layer_outputs, self.readouts)):
            readout_outputs.append(readout(out))
        
        # Ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        output = sum(w * r for w, r in zip(weights, readout_outputs))
        output = output / (self.output_temp.abs() + 0.1)
        
        return output, all_spikes
