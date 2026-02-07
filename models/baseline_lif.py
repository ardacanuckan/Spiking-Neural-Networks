"""
Baseline LIF Network
====================

Standard Leaky Integrate-and-Fire network for baseline comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .neurons import LIFNeuron


class BaselineLIF(nn.Module):
    """
    Baseline LIF Network for classification.
    
    Architecture:
        Input -> FC1 -> BN -> LIF -> FC2 -> BN -> LIF -> Output
    
    Args:
        input_size: Input dimension (default: 784 for MNIST)
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        tau: LIF time constant
    """
    
    def __init__(
        self,
        input_size=784,
        hidden_sizes=[512, 256],
        num_classes=10,
        tau=2.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.tau = tau
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(LIFNeuron(tau=tau))
            prev_size = hidden_size
        
        self.features = nn.ModuleList(layers)
        self.classifier = nn.Linear(prev_size, num_classes)
        
        # For spike counting
        self.spike_layers = [l for l in self.features if isinstance(l, LIFNeuron)]
    
    def reset_state(self):
        """Reset all neuron states."""
        for layer in self.features:
            if isinstance(layer, LIFNeuron):
                layer.reset_state()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, input_size] or [batch, C, H, W]
        
        Returns:
            output: Class logits [batch, num_classes]
            spikes: List of spike tensors from each LIF layer
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        spikes = []
        
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, LIFNeuron):
                spikes.append(x)
        
        output = self.classifier(x)
        
        return output, spikes
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineLIF_Conv(nn.Module):
    """
    Convolutional Baseline LIF for image classification.
    
    Architecture:
        Conv1 -> BN -> LIF -> Pool ->
        Conv2 -> BN -> LIF -> Pool ->
        Conv3 -> BN -> LIF -> Pool ->
        FC -> LIF -> Output
    """
    
    def __init__(
        self,
        in_channels=1,
        num_classes=10,
        tau=2.0
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = LIFNeuron(tau)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = LIFNeuron(tau)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = LIFNeuron(tau)
        
        self.pool = nn.AvgPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4)
        
        self.fc = nn.Linear(128 * 4 * 4, 256)
        self.lif_fc = LIFNeuron(tau)
        self.classifier = nn.Linear(256, num_classes)
    
    def reset_state(self):
        self.lif1.reset_state()
        self.lif2.reset_state()
        self.lif3.reset_state()
        self.lif_fc.reset_state()
    
    def forward(self, x):
        spikes = []
        
        # Conv block 1
        h = self.bn1(self.conv1(x))
        h = self.lif1(h)
        spikes.append(h)
        h = self.pool(h)
        
        # Conv block 2
        h = self.bn2(self.conv2(h))
        h = self.lif2(h)
        spikes.append(h)
        h = self.pool(h)
        
        # Conv block 3
        h = self.bn3(self.conv3(h))
        h = self.lif3(h)
        spikes.append(h)
        h = self.adaptive_pool(h)
        
        # FC
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        h = self.lif_fc(h)
        spikes.append(h)
        
        output = self.classifier(h)
        
        return output, spikes
