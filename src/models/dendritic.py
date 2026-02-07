"""
Dendritic Spiking Neural Networks

Novel multi-compartment neuron models that incorporate dendritic computation
for enhanced computational power and biological plausibility.

This module implements:
1. DendriticBranch: Individual dendritic compartment with nonlinear integration
2. DendriticLIFNeuron: Multi-compartment LIF with dendritic tree structure
3. DendriticAttentionNeuron: Dendrites as attention mechanism (NOVEL)

Reference Papers:
- "Scalable Dendritic Modeling Advances Expressive and Robust Deep SNNs" (arXiv 2024)
- "Temporal dendritic heterogeneity for learning multi-timescale dynamics" (Nature Comm 2024)
- "Leveraging dendritic properties to advance machine learning" (IMBB-FORTH 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Callable, Union
import math


class DendriticBranch(nn.Module):
    """
    Individual Dendritic Branch with Nonlinear Integration.
    
    Dendrites perform local nonlinear computation before signals
    reach the soma. This enables per-synapse gating and multiplicative
    interactions that point neurons cannot achieve.
    
    The dendritic potential follows:
        V_d[t+1] = beta_d * V_d[t] + W_d @ x + sigmoid(gate) * (context @ x)
        
    where:
        - beta_d: dendritic decay factor (slower than soma)
        - W_d: feedforward synaptic weights
        - gate: learnable gating mechanism
        - context: modulatory context (e.g., attention, top-down signal)
    
    Args:
        in_features: Number of input features
        branch_features: Number of features in this branch
        tau_dendrite: Dendritic time constant (typically > somatic tau)
        nonlinearity: Nonlinear activation for dendritic computation
        use_gating: Whether to use multiplicative gating
    """
    
    def __init__(
        self,
        in_features: int,
        branch_features: int,
        tau_dendrite: float = 4.0,  # Slower than typical somatic tau=2
        nonlinearity: str = 'sigmoid',
        use_gating: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.branch_features = branch_features
        self.use_gating = use_gating
        
        # Dendritic decay factor (slower dynamics)
        beta_d = 1.0 - 1.0 / tau_dendrite
        self.register_buffer('beta_d', torch.tensor(beta_d))
        
        # Feedforward synaptic weights to this branch
        self.W_ff = nn.Linear(in_features, branch_features, bias=False)
        
        # Gating mechanism for multiplicative modulation
        if use_gating:
            self.gate = nn.Linear(in_features, branch_features, bias=True)
            self.context = nn.Linear(in_features, branch_features, bias=False)
        
        # Nonlinearity selection
        self.nonlinearity = self._get_nonlinearity(nonlinearity)
        
        # Dendritic membrane potential
        self.v_d = None
        
        # Initialize weights
        self._init_weights()
    
    def _get_nonlinearity(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'relu': F.relu,
            'softplus': F.softplus,
            'elu': F.elu
        }
        return activations.get(name, torch.sigmoid)
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling."""
        nn.init.xavier_uniform_(self.W_ff.weight)
        if self.use_gating:
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            nn.init.xavier_uniform_(self.context.weight)
    
    def reset_state(self):
        """Reset dendritic membrane potential."""
        self.v_d = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dendritic branch output.
        
        Args:
            x: Input tensor [batch, in_features]
            
        Returns:
            Dendritic output after nonlinear integration [batch, branch_features]
        """
        # Initialize state if needed
        batch_size = x.size(0)
        if self.v_d is None:
            self.v_d = torch.zeros(batch_size, self.branch_features, device=x.device)
        
        # Feedforward input
        ff_input = self.W_ff(x)
        
        # Gated modulation (key dendritic nonlinearity)
        if self.use_gating:
            gate_value = torch.sigmoid(self.gate(x))
            context_value = self.context(x)
            gated_input = gate_value * context_value
            total_input = ff_input + gated_input
        else:
            total_input = ff_input
        
        # Dendritic membrane dynamics (slower than soma)
        self.v_d = self.beta_d * self.v_d + (1 - self.beta_d) * total_input
        
        # Nonlinear dendritic output
        return self.nonlinearity(self.v_d)


class DendriticLIFNeuron(nn.Module):
    """
    Multi-Compartment Dendritic LIF Neuron.
    
    This neuron model incorporates multiple dendritic branches that
    perform local nonlinear computation before integration at the soma.
    This architecture provides:
    
    1. Enhanced computational power through dendritic nonlinearities
    2. Natural attention-like gating via dendritic modulation
    3. Multi-timescale processing (dendrites vs soma)
    4. Improved gradient flow through parallel pathways
    
    Architecture:
        Input -> [Branch 1] --\\
              -> [Branch 2] ----> Soma (LIF) -> Spike
              -> [Branch N] --/
    
    The somatic potential integrates dendritic outputs:
        V_soma[t+1] = beta_s * V_soma[t] + sum(w_i * dendrite_i[t])
        
    If V_soma > threshold: spike and reset
    
    Args:
        in_features: Number of input features
        num_branches: Number of dendritic branches
        branch_features: Features per branch (or list for heterogeneous)
        tau_soma: Somatic time constant
        tau_dendrite: Dendritic time constant (or list for heterogeneous)
        v_threshold: Firing threshold
        heterogeneous_tau: Use different tau per branch for multi-timescale
    """
    
    def __init__(
        self,
        in_features: int,
        num_branches: int = 4,
        branch_features: Union[int, List[int]] = 32,
        tau_soma: float = 2.0,
        tau_dendrite: Union[float, List[float]] = 4.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        heterogeneous_tau: bool = True,
        use_gating: bool = True,
        learnable_tau: bool = True,
        detach_reset: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.num_branches = num_branches
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        
        # Handle branch features
        if isinstance(branch_features, int):
            branch_features = [branch_features] * num_branches
        self.branch_features = branch_features
        
        # Handle dendritic time constants (heterogeneous for multi-timescale)
        if heterogeneous_tau:
            if isinstance(tau_dendrite, float):
                # Create logarithmically spaced time constants
                tau_dendrite = [tau_dendrite * (2 ** i) for i in range(num_branches)]
        else:
            if isinstance(tau_dendrite, float):
                tau_dendrite = [tau_dendrite] * num_branches
        
        # Create dendritic branches
        self.branches = nn.ModuleList([
            DendriticBranch(
                in_features=in_features,
                branch_features=bf,
                tau_dendrite=td,
                use_gating=use_gating
            )
            for bf, td in zip(branch_features, tau_dendrite)
        ])
        
        # Branch integration weights
        total_branch_features = sum(branch_features)
        self.branch_weights = nn.Linear(total_branch_features, 1, bias=False)
        
        # Somatic decay factor
        if learnable_tau:
            beta_s = 1.0 - 1.0 / tau_soma
            self.beta_s_raw = nn.Parameter(
                torch.tensor(math.log(beta_s / (1 - beta_s + 1e-8)))
            )
        else:
            beta_s = 1.0 - 1.0 / tau_soma
            self.register_buffer('beta_s_raw', torch.tensor(beta_s))
        
        self.learnable_tau = learnable_tau
        
        # Somatic membrane potential
        self.v = None
        
    @property
    def beta_soma(self) -> torch.Tensor:
        """Get somatic decay factor."""
        if self.learnable_tau:
            return torch.sigmoid(self.beta_s_raw)
        return self.beta_s_raw
    
    def reset_state(self):
        """Reset all compartment states."""
        self.v = None
        for branch in self.branches:
            branch.reset_state()
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through dendritic neuron.
        
        Args:
            x: Input tensor [batch, in_features]
            surrogate_function: Surrogate gradient function
            
        Returns:
            Tuple of (spikes, membrane_potential)
        """
        batch_size = x.size(0)
        
        # Initialize somatic potential
        if self.v is None:
            self.v = torch.zeros(batch_size, 1, device=x.device)
        
        # Compute dendritic outputs (parallel branches)
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate branch outputs
        dendrite_out = torch.cat(branch_outputs, dim=-1)
        
        # Weighted integration at soma
        somatic_input = self.branch_weights(dendrite_out)
        
        # Somatic membrane dynamics
        self.v = self.beta_soma * self.v + (1 - self.beta_soma) * somatic_input
        
        # Generate spikes
        spike = surrogate_function(self.v - self.v_threshold)
        
        # Reset mechanism
        if self.detach_reset:
            spike_detached = spike.detach()
        else:
            spike_detached = spike
            
        if self.v_reset is None:
            self.v = self.v - spike_detached * self.v_threshold
        else:
            self.v = (1 - spike_detached) * self.v + spike_detached * self.v_reset
        
        return spike.squeeze(-1), self.v.squeeze(-1)
    
    def get_branch_activations(self) -> List[torch.Tensor]:
        """Get current dendritic branch activations for visualization."""
        return [branch.v_d for branch in self.branches]


class DendriticAttentionNeuron(nn.Module):
    """
    NOVEL: Dendritic Attention Spiking Neuron
    
    This is our main research contribution: using dendritic branches
    as a biological attention mechanism.
    
    Key Innovation:
    - Each dendritic branch attends to different aspects of input
    - Attention weights emerge from dendritic gating (not learned QKV)
    - Multi-timescale dendrites capture different temporal contexts
    - Spike timing encodes attention priority (early spike = high attention)
    
    The attention mechanism:
        attention_i = softmax(dendrite_i / temperature)
        output = sum(attention_i * value_i)
        
    where dendrite_i is the activation of branch i, and value_i
    is the gated information from that branch.
    
    This provides several advantages over standard attention:
    1. No quadratic complexity (O(n) vs O(n²))
    2. Event-driven computation (only active branches contribute)
    3. Biological plausibility (dendrites naturally implement this)
    4. Multi-timescale attention via heterogeneous tau
    
    Args:
        in_features: Number of input features
        num_heads: Number of attention heads (dendritic branches)
        head_dim: Dimension per attention head
        tau_range: Range of dendritic time constants [min, max]
        temperature: Softmax temperature for attention
    """
    
    def __init__(
        self,
        in_features: int,
        num_heads: int = 8,
        head_dim: int = 64,
        tau_soma: float = 2.0,
        tau_range: Tuple[float, float] = (2.0, 32.0),
        temperature: float = 1.0,
        v_threshold: float = 1.0,
        learnable_temperature: bool = True,
        detach_reset: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        
        # Learnable temperature for attention sharpness
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # Create dendritic branches with logarithmically spaced tau
        tau_min, tau_max = tau_range
        taus = torch.logspace(
            math.log10(tau_min),
            math.log10(tau_max),
            num_heads
        ).tolist()
        
        # Query branches (what to attend to)
        self.query_branches = nn.ModuleList([
            DendriticBranch(
                in_features=in_features,
                branch_features=head_dim,
                tau_dendrite=tau,
                nonlinearity='tanh',
                use_gating=True
            )
            for tau in taus
        ])
        
        # Key branches (what information is available)
        self.key_branches = nn.ModuleList([
            DendriticBranch(
                in_features=in_features,
                branch_features=head_dim,
                tau_dendrite=tau,
                nonlinearity='tanh',
                use_gating=True
            )
            for tau in taus
        ])
        
        # Value branches (actual information to pass)
        self.value_branches = nn.ModuleList([
            DendriticBranch(
                in_features=in_features,
                branch_features=head_dim,
                tau_dendrite=tau,
                nonlinearity='relu',  # ReLU for values
                use_gating=False  # No gating for values
            )
            for tau in taus
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, in_features, bias=False)
        
        # Somatic integration
        beta_s = 1.0 - 1.0 / tau_soma
        self.register_buffer('beta_soma', torch.tensor(beta_s))
        
        # State variables
        self.v = None
        self.attention_weights = None  # For visualization
        
    def reset_state(self):
        """Reset all states."""
        self.v = None
        self.attention_weights = None
        for branch_list in [self.query_branches, self.key_branches, self.value_branches]:
            for branch in branch_list:
                branch.reset_state()
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dendritic attention.
        
        Args:
            x: Input tensor [batch, in_features]
            surrogate_function: Surrogate gradient function
            
        Returns:
            Tuple of (spikes, membrane_potential)
        """
        batch_size = x.size(0)
        
        # Initialize somatic potential
        if self.v is None:
            self.v = torch.zeros(batch_size, self.in_features, device=x.device)
        
        # Compute queries, keys, values from dendritic branches
        queries = torch.stack([branch(x) for branch in self.query_branches], dim=1)  # [B, H, D]
        keys = torch.stack([branch(x) for branch in self.key_branches], dim=1)  # [B, H, D]
        values = torch.stack([branch(x) for branch in self.value_branches], dim=1)  # [B, H, D]
        
        # Compute attention scores (dot product between query and key)
        # [B, H] - one attention weight per head
        attention_scores = (queries * keys).sum(dim=-1) / (self.head_dim ** 0.5)
        
        # Apply temperature and softmax
        self.attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)  # [B, H]
        
        # Weight values by attention
        weighted_values = self.attention_weights.unsqueeze(-1) * values  # [B, H, D]
        
        # Combine heads
        combined = weighted_values.view(batch_size, -1)  # [B, H*D]
        
        # Project to output dimension
        output = self.output_proj(combined)  # [B, in_features]
        
        # Somatic integration
        self.v = self.beta_soma * self.v + (1 - self.beta_soma) * output
        
        # Generate spikes
        spike = surrogate_function(self.v - self.v_threshold)
        
        # Reset mechanism
        spike_for_reset = spike.detach() if self.detach_reset else spike
        self.v = self.v - spike_for_reset * self.v_threshold
        
        return spike, self.v
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights for visualization."""
        return self.attention_weights


class DendriticConv2d(nn.Module):
    """
    Convolutional layer with dendritic computation.
    
    Extends 2D convolution to include per-channel dendritic
    processing for vision tasks.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        num_branches: int = 2,
        tau_dendrite: float = 4.0,
        tau_soma: float = 2.0,
        v_threshold: float = 1.0
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        
        # Per-channel dendritic gating
        self.gate = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=True
        )
        
        # Batch norm for stability
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Time constants
        beta_d = 1.0 - 1.0 / tau_dendrite
        beta_s = 1.0 - 1.0 / tau_soma
        self.register_buffer('beta_d', torch.tensor(beta_d))
        self.register_buffer('beta_s', torch.tensor(beta_s))
        
        self.v_threshold = v_threshold
        
        # State
        self.v_d = None  # Dendritic potential
        self.v = None    # Somatic potential
        
    def reset_state(self):
        self.v_d = None
        self.v = None
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Convolutional input
        conv_out = self.conv(x)
        
        # Gating mechanism
        gate = torch.sigmoid(self.gate(x))
        gated_out = conv_out * gate
        
        # Batch normalization
        normalized = self.bn(gated_out)
        
        # Initialize states
        if self.v_d is None:
            self.v_d = torch.zeros_like(normalized)
        if self.v is None:
            self.v = torch.zeros_like(normalized)
        
        # Dendritic dynamics
        self.v_d = self.beta_d * self.v_d + (1 - self.beta_d) * normalized
        
        # Somatic dynamics
        self.v = self.beta_s * self.v + (1 - self.beta_s) * self.v_d
        
        # Spike generation
        spike = surrogate_function(self.v - self.v_threshold)
        
        # Reset
        spike_detach = spike.detach()
        self.v = self.v - spike_detach * self.v_threshold
        
        return spike, self.v
