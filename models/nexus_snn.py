"""
NEXUS-SNN: Neural EXpressive Unified Spiking Neural Network
============================================================

Combines multiple SOTA techniques:
- Chebyshev KAN activations
- Adaptive thresholds
- Temporal attention
- Residual connections
- Heterogeneous time constants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .surrogate import spike_fn
from .neurons import AdaptiveLIF


class ChebyshevKAN(nn.Module):
    """
    KAN activation using Chebyshev polynomials.
    More numerically stable than standard polynomials.
    """
    
    def __init__(self, features, degree=4):
        super().__init__()
        
        self.features = features
        self.degree = degree
        
        # Learnable coefficients
        self.coefficients = nn.Parameter(torch.zeros(features, degree + 1))
        nn.init.constant_(self.coefficients[:, 1], 1.0)  # Identity
        nn.init.normal_(self.coefficients[:, 2:], 0, 0.01)
        
        # Output scale
        self.scale = nn.Parameter(torch.ones(features))
        
        # Residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, x):
        x_norm = torch.tanh(x * 0.3)
        
        # Chebyshev recurrence
        T0 = torch.ones_like(x_norm)
        T1 = x_norm
        
        result = self.coefficients[:, 0].unsqueeze(0) * T0
        if self.degree >= 1:
            result = result + self.coefficients[:, 1].unsqueeze(0) * T1
        
        T_prev, T_curr = T0, T1
        for i in range(2, self.degree + 1):
            T_next = 2 * x_norm * T_curr - T_prev
            result = result + self.coefficients[:, i].unsqueeze(0) * T_next
            T_prev, T_curr = T_curr, T_next
        
        return self.scale.unsqueeze(0) * result + self.residual_weight * x


class TemporalAttention(nn.Module):
    """
    Learns optimal weighting of time steps.
    """
    
    def __init__(self, hidden_size, max_time=10):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, outputs):
        """
        Args:
            outputs: List of [batch, features] tensors
        """
        if len(outputs) == 1:
            return outputs[0]
        
        stacked = torch.stack(outputs, dim=1)  # [batch, time, features]
        scores = self.attention(stacked).squeeze(-1)  # [batch, time]
        weights = F.softmax(scores, dim=1)
        
        return (weights.unsqueeze(-1) * stacked).sum(dim=1)


class NEXUSLayer(nn.Module):
    """
    Single NEXUS layer combining all innovations.
    """
    
    def __init__(self, in_features, out_features, tau=2.0, dropout=0.1):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.kan = ChebyshevKAN(out_features, degree=4)
        self.dropout = nn.Dropout(dropout)
        self.lif = AdaptiveLIF(out_features, tau=tau)
    
    def reset_state(self):
        self.lif.reset_state()
    
    def forward(self, x):
        h = self.bn(self.linear(x))
        h = self.kan(h)
        h = self.dropout(h)
        spike = self.lif(h)
        return spike, self.lif.v_mem


class NEXUSSNN(nn.Module):
    """
    NEXUS-SNN: Ultimate model combining all innovations.
    
    Architecture:
        Input -> [NEXUS Layers with heterogeneous tau] -> Temporal Attention -> Output
    
    Args:
        input_size: Input dimension
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        tau_range: Range of time constants (heterogeneous)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_size=784,
        hidden_sizes=[512, 256],
        num_classes=10,
        tau_range=(1.5, 6.0),
        dropout=0.15
    ):
        super().__init__()
        
        self.hidden_sizes = hidden_sizes
        
        # Heterogeneous time constants
        taus = np.linspace(tau_range[0], tau_range[1], len(hidden_sizes) + 1)
        
        # Input layer
        self.input_linear = nn.Linear(input_size, hidden_sizes[0])
        self.input_bn = nn.BatchNorm1d(hidden_sizes[0])
        self.input_lif = AdaptiveLIF(hidden_sizes[0], tau=taus[0], base_threshold=1.0)
        
        # NEXUS layers
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(
                NEXUSLayer(hidden_sizes[i], hidden_sizes[i+1], tau=taus[i+1], dropout=dropout)
            )
        
        # Residual projection
        if len(hidden_sizes) > 1:
            self.residual_proj = nn.Linear(hidden_sizes[0], hidden_sizes[-1])
            self.residual_weight = nn.Parameter(torch.tensor(0.3))
        
        # Output
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Temporal attention
        self.temporal_attn = TemporalAttention(num_classes, max_time=10)
    
    def reset_state(self):
        self.input_lif.reset_state()
        for layer in self.layers:
            layer.reset_state()
    
    def forward(self, x, time_step=0):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Input encoding
        h = self.input_bn(self.input_linear(x))
        spike = self.input_lif(h)
        first_spike = spike
        
        all_spikes = [spike]
        membrane_potentials = [self.input_lif.v_mem]
        
        # NEXUS layers
        for layer in self.layers:
            spike, v_mem = layer(spike)
            all_spikes.append(spike)
            membrane_potentials.append(v_mem)
        
        # Residual
        if hasattr(self, 'residual_proj'):
            residual = self.residual_proj(first_spike)
            spike = spike + self.residual_weight.abs() * residual
        
        # Output
        out = self.output_layer(spike)
        
        return out, all_spikes, membrane_potentials
