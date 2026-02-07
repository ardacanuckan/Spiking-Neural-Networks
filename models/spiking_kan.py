"""
Spiking-KAN: Spiking Kolmogorov-Arnold Network
==============================================

First implementation combining KAN's learnable activation functions 
with spiking neuron dynamics.

Key innovations:
- Learnable polynomial basis functions
- Spike generation from KAN outputs
- Interpretable learned activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .surrogate import spike_fn
from .neurons import LIFNeuron


class KANActivation(nn.Module):
    """
    Kolmogorov-Arnold Network activation.
    
    Implements learnable activation functions using polynomial basis:
        phi(x) = sum_i c_i * B_i(x)
    
    where B_i are basis functions (powers, Chebyshev, etc.)
    """
    
    def __init__(self, in_features, out_features, degree=4, basis='chebyshev'):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.basis = basis
        
        # Learnable coefficients: [out, in, degree+1]
        self.coefficients = nn.Parameter(torch.zeros(out_features, in_features, degree + 1))
        
        # Initialize: linear term = 1, others = small
        nn.init.constant_(self.coefficients[:, :, 1], 1.0 / in_features)
        nn.init.normal_(self.coefficients[:, :, 2:], 0, 0.01)
        
        # Output scaling
        self.scale = nn.Parameter(torch.ones(out_features))
    
    def _chebyshev_basis(self, x, degree):
        """Compute Chebyshev polynomial basis."""
        # Normalize to [-1, 1]
        x_norm = torch.tanh(x * 0.5)
        
        # T0 = 1, T1 = x, Tn = 2x*Tn-1 - Tn-2
        T = [torch.ones_like(x_norm), x_norm]
        for _ in range(2, degree + 1):
            T.append(2 * x_norm * T[-1] - T[-2])
        
        return torch.stack(T, dim=-1)  # [batch, in_features, degree+1]
    
    def _power_basis(self, x, degree):
        """Compute power polynomial basis."""
        powers = torch.arange(degree + 1, device=x.device, dtype=x.dtype)
        return x.unsqueeze(-1).pow(powers)  # [batch, in_features, degree+1]
    
    def forward(self, x):
        """
        Args:
            x: [batch, in_features]
        Returns:
            [batch, out_features]
        """
        # Get basis functions
        if self.basis == 'chebyshev':
            basis = self._chebyshev_basis(x, self.degree)
        else:
            basis = self._power_basis(x, self.degree)
        
        # Apply learned coefficients: sum over in_features and degree
        # [batch, in, degree+1] @ [out, in, degree+1] -> [batch, out]
        output = torch.einsum('bid,oid->bo', basis, self.coefficients)
        
        return self.scale * output


class SpikingKANLayer(nn.Module):
    """
    Single Spiking-KAN layer.
    
    Combines:
    - KAN activation (learnable polynomial transformation)
    - Batch normalization
    - LIF spiking neuron
    """
    
    def __init__(self, in_features, out_features, tau=2.0, degree=4):
        super().__init__()
        
        self.kan = KANActivation(in_features, out_features, degree, basis='chebyshev')
        self.bn = nn.BatchNorm1d(out_features)
        self.lif = LIFNeuron(tau=tau)
    
    def reset_state(self):
        self.lif.reset_state()
    
    def forward(self, x):
        h = self.kan(x)
        h = self.bn(h)
        spike = self.lif(h)
        return spike


class SpikingKAN(nn.Module):
    """
    Spiking Kolmogorov-Arnold Network.
    
    Architecture:
        Input -> KANLayer1 -> KANLayer2 -> ... -> Output
    
    Args:
        input_size: Input dimension
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        tau: LIF time constant
        degree: Polynomial degree for KAN
    """
    
    def __init__(
        self,
        input_size=784,
        hidden_sizes=[512, 256],
        num_classes=10,
        tau=2.0,
        degree=4
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(SpikingKANLayer(prev_size, hidden_size, tau, degree))
            prev_size = hidden_size
        
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(prev_size, num_classes)
    
    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        spikes = []
        
        for layer in self.layers:
            x = layer(x)
            spikes.append(x)
        
        output = self.classifier(x)
        
        return output, spikes
    
    def visualize_activations(self, layer_idx=0):
        """
        Visualize learned activation functions.
        Returns x values and corresponding phi(x) for plotting.
        """
        layer = self.layers[layer_idx]
        kan = layer.kan
        
        x = torch.linspace(-3, 3, 100).unsqueeze(0)
        x = x.expand(kan.in_features, -1).T  # [100, in_features]
        
        with torch.no_grad():
            y = kan(x)
        
        return x.numpy(), y.numpy()
