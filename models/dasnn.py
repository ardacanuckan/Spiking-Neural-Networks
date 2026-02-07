"""
DASNN: Dendritic Attention Spiking Neural Network
=================================================

Multi-compartment dendritic neurons with:
- Heterogeneous time constants
- Gating mechanisms  
- Attention via dendritic competition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .surrogate import spike_fn


class DendriticNeuron(nn.Module):
    """
    Multi-compartment dendritic neuron.
    
    Each neuron has multiple dendritic branches with different time constants.
    The soma integrates inputs from all branches with learned weights.
    """
    
    def __init__(self, in_features, out_features, num_branches=4, tau_range=(1.5, 8.0)):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_branches = num_branches
        
        # Heterogeneous time constants
        taus = np.linspace(tau_range[0], tau_range[1], num_branches)
        self.register_buffer('betas', torch.tensor([1.0 - 1.0/t for t in taus], dtype=torch.float32))
        
        # Dendritic branch weights
        self.branch_weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, out_features) * 0.1)
            for _ in range(num_branches)
        ])
        
        # Somatic integration weights
        self.soma_weights = nn.Parameter(torch.ones(num_branches) / num_branches)
        
        # Threshold
        self.threshold = nn.Parameter(torch.ones(out_features))
        
        # States
        self.branch_v = None
        self.soma_v = None
    
    def reset_state(self):
        self.branch_v = None
        self.soma_v = None
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize states
        if self.branch_v is None:
            self.branch_v = [torch.zeros(batch_size, self.out_features, device=x.device) 
                           for _ in range(self.num_branches)]
            self.soma_v = torch.zeros(batch_size, self.out_features, device=x.device)
        
        # Process each dendritic branch
        branch_outputs = []
        for i, (beta, w) in enumerate(zip(self.betas, self.branch_weights)):
            # Dendritic input
            current = torch.matmul(x, w)
            
            # Leaky integration with branch-specific tau
            self.branch_v[i] = beta * self.branch_v[i] + (1 - beta) * current
            branch_outputs.append(self.branch_v[i])
        
        # Soma integration (weighted sum of branches)
        soma_weights = F.softmax(self.soma_weights, dim=0)
        self.soma_v = sum(w * b for w, b in zip(soma_weights, branch_outputs))
        
        # Spike generation
        spike = spike_fn(self.soma_v - self.threshold)
        
        # Reset
        self.soma_v = self.soma_v - spike.detach() * self.threshold
        
        return spike


class DendriticAttention(nn.Module):
    """
    Attention mechanism via dendritic competition.
    Implements O(n) complexity attention using dendritic dynamics.
    """
    
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query, Key projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dendritic gating
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
    
    def forward(self, x, context=None):
        """
        Args:
            x: Input features [batch, features]
            context: Optional context [batch, features]
        """
        if context is None:
            context = x
        
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        # Dendritic gating
        gate_input = torch.cat([x, context], dim=-1)
        gate = self.gate(gate_input)
        
        # Simple attention (element-wise for efficiency)
        attention = torch.sigmoid(q * k / np.sqrt(self.hidden_size))
        out = attention * v * gate
        
        return self.out_proj(out)


class DASNN(nn.Module):
    """
    Dendritic Attention Spiking Neural Network.
    
    Architecture:
        Input -> Encoder -> DendriticLayer1 -> DendriticLayer2 -> Attention -> Classifier
    
    Args:
        input_size: Input dimension
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        num_branches: Number of dendritic branches per neuron
        tau_range: Range of time constants
    """
    
    def __init__(
        self,
        input_size=784,
        hidden_sizes=[512, 256],
        num_classes=10,
        num_branches=4,
        tau_range=(1.5, 8.0)
    ):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0])
        )
        
        # Dendritic layers
        self.dendritic_layers = nn.ModuleList()
        prev_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            self.dendritic_layers.append(
                DendriticNeuron(prev_size, hidden_size, num_branches, tau_range)
            )
            prev_size = hidden_size
        
        # Attention
        self.attention = DendriticAttention(hidden_sizes[-1])
        
        # Classifier
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)
    
    def reset_state(self):
        for layer in self.dendritic_layers:
            layer.reset_state()
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        spikes = []
        
        # Encode
        h = self.encoder(x)
        
        # Dendritic processing
        for layer in self.dendritic_layers:
            h = layer(h)
            spikes.append(h)
        
        # Attention
        h = self.attention(h)
        
        # Classify
        output = self.classifier(h)
        
        return output, spikes
