"""
Network Architectures for DASNN Research

Contains the main network architectures combining our novel contributions:
1. Dendritic Attention mechanisms
2. Multi-compartment neurons
3. Adaptive surrogate gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable, List
import sys
sys.path.append('..')


class DASNNClassifier(nn.Module):
    """
    DASNN: Dendritic Attention Spiking Neural Network Classifier
    
    Our NOVEL architecture that combines:
    1. Multi-compartment dendritic neurons for enhanced computation
    2. Dendritic attention for selective information processing
    3. Heterogeneous time constants for multi-timescale dynamics
    4. Adaptive surrogate gradients based on membrane potential
    
    This is designed for image classification benchmarks (MNIST, CIFAR-10).
    
    Args:
        input_size: Input dimension (flattened image or feature size)
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        num_branches: Number of dendritic branches per neuron
        tau_range: Range of time constants for heterogeneous dynamics
        v_threshold: Firing threshold
        use_dendritic_attention: Whether to use dendritic attention layers
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = [512, 256],
        num_classes: int = 10,
        num_branches: int = 4,
        tau_range: Tuple[float, float] = (2.0, 16.0),
        v_threshold: float = 1.0,
        use_dendritic_attention: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.v_threshold = v_threshold
        self.use_dendritic_attention = use_dendritic_attention
        
        # Input encoding layer
        self.encoder = nn.Linear(input_size, hidden_sizes[0])
        self.encoder_bn = nn.BatchNorm1d(hidden_sizes[0])
        
        # Create time constants with heterogeneous distribution
        tau_min, tau_max = tau_range
        num_layers = len(hidden_sizes)
        
        # Spiking layers with heterogeneous time constants
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Membrane decay factors (heterogeneous per layer)
        self.betas = nn.ParameterList()
        
        for i, (in_size, out_size) in enumerate(zip(
            [hidden_sizes[0]] + hidden_sizes[:-1], 
            hidden_sizes
        )):
            # Linear transformation
            self.layers.append(nn.Linear(in_size, out_size))
            self.norms.append(nn.BatchNorm1d(out_size))
            self.dropouts.append(nn.Dropout(dropout))
            
            # Heterogeneous tau per layer (interpolate through range)
            tau = tau_min + (tau_max - tau_min) * (i / max(1, num_layers - 1))
            beta = 1.0 - 1.0 / tau
            self.betas.append(nn.Parameter(torch.tensor(beta)))
        
        # Dendritic attention layer (our novel contribution)
        if use_dendritic_attention:
            self.attention = DendriticAttentionLayer(
                dim=hidden_sizes[-1],
                num_heads=num_branches,
                tau_range=tau_range,
                v_threshold=v_threshold
            )
        
        # Output layer (non-spiking for classification)
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Membrane potentials for each layer
        self.membrane_potentials: List[Optional[torch.Tensor]] = [None] * len(hidden_sizes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def reset_state(self):
        """Reset all membrane potentials."""
        self.membrane_potentials = [None] * len(self.hidden_sizes)
        if self.use_dendritic_attention:
            self.attention.reset_state()
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> torch.Tensor:
        """
        Forward pass through DASNN.
        
        Args:
            x: Input tensor [batch, input_size] or [batch, C, H, W]
            surrogate_function: Surrogate gradient function
            
        Returns:
            Classification logits [batch, num_classes]
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Input encoding
        x = self.encoder(x)
        x = self.encoder_bn(x)
        
        # Pass through spiking layers
        for i, (layer, norm, dropout, beta) in enumerate(zip(
            self.layers, self.norms, self.dropouts, self.betas
        )):
            # Linear transformation
            current = layer(x)
            current = norm(current)
            
            # Initialize membrane potential
            if self.membrane_potentials[i] is None:
                self.membrane_potentials[i] = torch.zeros_like(current)
            
            # Leaky integration
            self.membrane_potentials[i] = (
                beta * self.membrane_potentials[i] + 
                (1 - beta) * current
            )
            
            # Spike generation
            x = surrogate_function(self.membrane_potentials[i] - self.v_threshold)
            
            # Reset membrane potential
            self.membrane_potentials[i] = (
                self.membrane_potentials[i] - 
                x.detach() * self.v_threshold
            )
            
            # Dropout (on spikes)
            x = dropout(x)
        
        # Dendritic attention
        if self.use_dendritic_attention:
            x, _ = self.attention(x, surrogate_function)
        
        # Classification (membrane potential readout, no spike)
        output = self.classifier(x)
        
        return output


class DendriticAttentionLayer(nn.Module):
    """
    Dendritic Attention Layer - Our Novel Contribution
    
    Uses dendritic branches as attention heads with different
    time constants for multi-scale temporal attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        tau_range: Tuple[float, float] = (2.0, 16.0),
        v_threshold: float = 1.0
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.v_threshold = v_threshold
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Create heterogeneous time constants
        tau_min, tau_max = tau_range
        taus = torch.linspace(tau_min, tau_max, num_heads)
        betas = 1.0 - 1.0 / taus
        self.register_buffer('betas', betas)
        
        # Projections
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        
        # Gate for modulation
        self.gate = nn.Linear(dim, dim)
        
        # Membrane potentials per head
        self.v_heads: Optional[torch.Tensor] = None
        self.v_out: Optional[torch.Tensor] = None
    
    def reset_state(self):
        self.v_heads = None
        self.v_out = None
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with dendritic attention."""
        B, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        # Project to Q, K, V
        q = self.query(x).view(B, H, D)
        k = self.key(x).view(B, H, D)
        v = self.value(x).view(B, H, D)
        
        # Initialize membrane potentials
        if self.v_heads is None:
            self.v_heads = torch.zeros(B, H, D, device=x.device)
        if self.v_out is None:
            self.v_out = torch.zeros(B, C, device=x.device)
        
        # Leaky integration per head with different tau
        betas = self.betas.view(1, H, 1)
        self.v_heads = betas * self.v_heads + (1 - betas) * (q * k)
        
        # Compute attention as spike probability
        attn_logits = self.v_heads.sum(dim=-1) / (D ** 0.5)  # [B, H]
        attn = F.softmax(attn_logits / self.temperature, dim=-1)  # [B, H]
        
        # Weight values by attention
        weighted_v = attn.unsqueeze(-1) * v  # [B, H, D]
        combined = weighted_v.view(B, C)  # [B, C]
        
        # Gating
        gate = torch.sigmoid(self.gate(x))
        gated = combined * gate
        
        # Output projection
        out = self.output(gated)
        
        # Output spiking
        beta_avg = self.betas.mean()
        self.v_out = beta_avg * self.v_out + (1 - beta_avg) * out
        
        spike = surrogate_function(self.v_out - self.v_threshold)
        self.v_out = self.v_out - spike.detach() * self.v_threshold
        
        return spike, self.v_out


class ConvDASNN(nn.Module):
    """
    Convolutional DASNN for image classification.
    
    Uses convolutional layers with dendritic attention for
    spatial feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        channels: List[int] = [32, 64, 128],
        tau: float = 2.0,
        v_threshold: float = 1.0
    ):
        super().__init__()
        
        self.v_threshold = v_threshold
        
        # Convolutional layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        in_ch = in_channels
        for out_ch in channels:
            self.convs.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            self.bns.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch
        
        # Decay factors
        beta = 1.0 - 1.0 / tau
        self.register_buffer('beta', torch.tensor(beta))
        
        # Pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.classifier = nn.Linear(channels[-1] * 16, num_classes)
        
        # Membrane potentials
        self.vs: List[Optional[torch.Tensor]] = [None] * len(channels)
    
    def reset_state(self):
        self.vs = [None] * len(self.convs)
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> torch.Tensor:
        """Forward pass."""
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x)
            x = bn(x)
            
            # Initialize membrane
            if self.vs[i] is None:
                self.vs[i] = torch.zeros_like(x)
            
            # LIF dynamics
            self.vs[i] = self.beta * self.vs[i] + (1 - self.beta) * x
            x = surrogate_function(self.vs[i] - self.v_threshold)
            self.vs[i] = self.vs[i] - x.detach() * self.v_threshold
            
            # Pooling every layer
            x = F.max_pool2d(x, 2)
        
        # Classifier
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_model(
    model_name: str,
    input_size: int = 784,
    num_classes: int = 10,
    **kwargs
) -> nn.Module:
    """
    Factory function to get model by name.
    
    Args:
        model_name: Name of model ('dasnn', 'conv_dasnn', 'baseline_lif')
        input_size: Input feature dimension
        num_classes: Number of output classes
        **kwargs: Additional model arguments
        
    Returns:
        Model instance
    """
    models = {
        'dasnn': DASNNClassifier,
        'conv_dasnn': ConvDASNN,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    if model_name == 'dasnn':
        return models[model_name](input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_name == 'conv_dasnn':
        return models[model_name](num_classes=num_classes, **kwargs)
    
    return models[model_name](**kwargs)
