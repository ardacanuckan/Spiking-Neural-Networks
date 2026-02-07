"""
Core Spiking Neural Network Layers

Building blocks for constructing deep SNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable, Union


class SpikingLinear(nn.Module):
    """
    Spiking Linear Layer with integrated LIF neuron.
    
    Combines a linear transformation with leaky integrate-and-fire
    dynamics in a single module for convenience.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        tau: Membrane time constant
        v_threshold: Firing threshold
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        bias: bool = True,
        detach_reset: bool = True
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LIF parameters
        beta = 1.0 - 1.0 / tau
        self.register_buffer('beta', torch.tensor(beta))
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        
        # State
        self.v = None
    
    def reset_state(self):
        self.v = None
    
    def forward(
        self, 
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, in_features] or spikes
            surrogate_function: Surrogate gradient function
            
        Returns:
            Tuple of (spikes, membrane_potential)
        """
        # Linear transformation
        current = self.linear(x)
        
        # Initialize membrane potential
        if self.v is None:
            self.v = torch.zeros_like(current)
        
        # Leaky integration
        self.v = self.beta * self.v + (1 - self.beta) * current
        
        # Spike generation
        spike = surrogate_function(self.v - self.v_threshold)
        
        # Reset
        spike_for_reset = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_for_reset * self.v_threshold
        else:
            self.v = (1 - spike_for_reset) * self.v + spike_for_reset * self.v_reset
        
        return spike, self.v


class SpikingConv2d(nn.Module):
    """
    Spiking 2D Convolution Layer with integrated LIF neuron.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding size
        tau: Membrane time constant
        v_threshold: Firing threshold
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        bias: bool = False,
        detach_reset: bool = True
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        
        # LIF parameters
        beta = 1.0 - 1.0 / tau
        self.register_buffer('beta', torch.tensor(beta))
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        
        self.v = None
    
    def reset_state(self):
        self.v = None
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        current = self.conv(x)
        
        if self.v is None:
            self.v = torch.zeros_like(current)
        
        self.v = self.beta * self.v + (1 - self.beta) * current
        
        spike = surrogate_function(self.v - self.v_threshold)
        
        spike_for_reset = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_for_reset * self.v_threshold
        else:
            self.v = (1 - spike_for_reset) * self.v + spike_for_reset * self.v_reset
        
        return spike, self.v


class SpikingBatchNorm2d(nn.Module):
    """
    Batch Normalization for Spiking Neural Networks.
    
    Applies batch normalization before the spiking nonlinearity,
    which helps stabilize training by normalizing the pre-spike
    membrane potential distribution.
    
    Uses Threshold-Dependent Batch Normalization (TDBN) which
    normalizes relative to the firing threshold.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        v_threshold: float = 1.0
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.v_threshold = v_threshold
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply threshold-aware batch normalization.
        
        The output is scaled such that the mean is at v_threshold/2,
        placing the distribution centered around the most gradient-active region.
        """
        normalized = self.bn(x)
        # Scale to place distribution around threshold
        return normalized * self.v_threshold / 2 + self.v_threshold / 2


class SpikingDropout(nn.Module):
    """
    Dropout for Spiking Neural Networks.
    
    Randomly zeros out entire spikes (neurons) during training.
    Unlike standard dropout, this operates on binary spike trains.
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout to spike trains."""
        if self.training:
            # Create mask and apply to spikes
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
            return x * mask / (1 - self.p)  # Scale to maintain expected value
        return x


class TemporalWiseDropout(nn.Module):
    """
    Dropout that varies across time steps.
    
    Different dropout patterns at each time step can improve
    temporal feature learning by preventing co-adaptation.
    """
    
    def __init__(self, p: float = 0.5, temporal_coherence: float = 0.5):
        super().__init__()
        self.p = p
        self.temporal_coherence = temporal_coherence
    
    def forward(self, x: torch.Tensor, time_step: int = 0) -> torch.Tensor:
        """
        Apply time-varying dropout.
        
        Args:
            x: Input tensor
            time_step: Current time step (used for varying pattern)
        """
        if not self.training:
            return x
        
        # Use different random seed per time step
        generator = torch.Generator(device=x.device)
        generator.manual_seed(time_step * 12345)
        
        mask = torch.bernoulli(
            torch.ones_like(x) * (1 - self.p),
            generator=generator
        )
        return x * mask / (1 - self.p)


class SpikingResidualBlock(nn.Module):
    """
    Residual block for spiking neural networks.
    
    Implements spike element-wise (SEW) residual connection:
    output = f(input) + input
    
    This preserves spike information across layers better than
    standard residual connections with continuous values.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        tau: float = 2.0,
        v_threshold: float = 1.0
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = SpikingConv2d(
            channels, channels, kernel_size,
            padding=padding, tau=tau, v_threshold=v_threshold
        )
        self.bn1 = SpikingBatchNorm2d(channels, v_threshold=v_threshold)
        
        self.conv2 = SpikingConv2d(
            channels, channels, kernel_size,
            padding=padding, tau=tau, v_threshold=v_threshold
        )
        self.bn2 = SpikingBatchNorm2d(channels, v_threshold=v_threshold)
    
    def reset_state(self):
        self.conv1.reset_state()
        self.conv2.reset_state()
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with residual connection."""
        identity = x
        
        # First conv block
        out = self.bn1(x)
        out, v1 = self.conv1(out, surrogate_function)
        
        # Second conv block
        out = self.bn2(out)
        out, v2 = self.conv2(out, surrogate_function)
        
        # Spike element-wise residual (OR operation for spikes)
        # This ensures at least one path carries information
        out = torch.max(out, identity)
        
        return out, v2
