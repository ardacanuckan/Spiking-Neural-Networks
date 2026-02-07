"""
Adaptive Spiking Neuron Models

Implements neurons with adaptive thresholds and learnable dynamics
that improve training stability and performance.

Reference Papers:
- "Rethinking the Membrane Dynamics and Optimization Objectives of SNNs" (NeurIPS 2024)
- "Adaptive Gradient Learning for SNNs by Exploiting MPD" (IJCAI 2025)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
import math


class AdaptiveLIFNeuron(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire Neuron with Dynamic Threshold.
    
    The threshold adapts based on recent activity, implementing
    intrinsic excitability modulation:
    
        theta[t+1] = theta_0 + beta_theta * (theta[t] - theta_0) + spike[t] * delta_theta
        
    This provides:
    1. Homeostatic regulation of firing rates
    2. Spike-frequency adaptation
    3. Better gradient flow through threshold dynamics
    
    Args:
        v_threshold_base: Base threshold value
        v_reset: Reset voltage
        tau_mem: Membrane time constant
        tau_thresh: Threshold adaptation time constant
        delta_thresh: Threshold increment per spike
        learnable_params: Whether tau and threshold params are learnable
    """
    
    def __init__(
        self,
        v_threshold_base: float = 1.0,
        v_reset: Optional[float] = 0.0,
        tau_mem: float = 2.0,
        tau_thresh: float = 10.0,
        delta_thresh: float = 0.1,
        learnable_params: bool = True,
        detach_reset: bool = True
    ):
        super().__init__()
        
        self.v_threshold_base = v_threshold_base
        self.v_reset = v_reset
        self.delta_thresh = delta_thresh
        self.detach_reset = detach_reset
        
        # Membrane decay
        beta_mem = 1.0 - 1.0 / tau_mem
        # Threshold decay (slower than membrane)
        beta_thresh = 1.0 - 1.0 / tau_thresh
        
        if learnable_params:
            self.beta_mem_raw = nn.Parameter(
                torch.tensor(math.log(beta_mem / (1 - beta_mem + 1e-8)))
            )
            self.beta_thresh_raw = nn.Parameter(
                torch.tensor(math.log(beta_thresh / (1 - beta_thresh + 1e-8)))
            )
            self.delta_thresh = nn.Parameter(torch.tensor(delta_thresh))
        else:
            self.register_buffer('beta_mem_raw', torch.tensor(beta_mem))
            self.register_buffer('beta_thresh_raw', torch.tensor(beta_thresh))
            self.register_buffer('delta_thresh_buf', torch.tensor(delta_thresh))
        
        self.learnable_params = learnable_params
        
        # State
        self.v = None
        self.theta = None  # Adaptive threshold
    
    @property
    def beta_mem(self) -> torch.Tensor:
        if self.learnable_params:
            return torch.sigmoid(self.beta_mem_raw)
        return self.beta_mem_raw
    
    @property
    def beta_thresh(self) -> torch.Tensor:
        if self.learnable_params:
            return torch.sigmoid(self.beta_thresh_raw)
        return self.beta_thresh_raw
    
    def reset_state(self):
        self.v = None
        self.theta = None
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with adaptive threshold.
        
        Args:
            x: Input current
            surrogate_function: Surrogate gradient function
            
        Returns:
            Tuple of (spikes, membrane_potential)
        """
        # Initialize states
        if self.v is None:
            self.v = torch.zeros_like(x)
        if self.theta is None:
            self.theta = torch.full_like(x, self.v_threshold_base)
        
        # Membrane dynamics
        self.v = self.beta_mem * self.v + (1 - self.beta_mem) * x
        
        # Generate spikes using adaptive threshold
        spike = surrogate_function(self.v - self.theta)
        
        # Update adaptive threshold
        delta = self.delta_thresh if self.learnable_params else self.delta_thresh_buf
        spike_for_adapt = spike.detach() if self.detach_reset else spike
        self.theta = (
            self.v_threshold_base + 
            self.beta_thresh * (self.theta - self.v_threshold_base) +
            spike_for_adapt * delta
        )
        
        # Reset membrane
        spike_for_reset = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_for_reset * self.theta
        else:
            self.v = (1 - spike_for_reset) * self.v + spike_for_reset * self.v_reset
        
        return spike, self.v
    
    def get_threshold(self) -> Optional[torch.Tensor]:
        """Get current adaptive threshold for visualization."""
        return self.theta


class LearnableIMPNeuron(nn.Module):
    """
    LIF Neuron with Learnable Initial Membrane Potential.
    
    Based on "Rethinking the Membrane Dynamics and Optimization Objectives of SNNs"
    which shows that setting IMP=0 limits spike pattern diversity.
    
    By learning the initial membrane potential, we enable:
    1. Additional firing patterns and pattern mappings
    2. Faster membrane potential evolution
    3. Accelerated training convergence
    
    Args:
        num_features: Number of neurons (for per-neuron IMP)
        v_threshold: Firing threshold
        tau: Membrane time constant
        init_imp: Initial membrane potential initialization
    """
    
    def __init__(
        self,
        num_features: int,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        tau: float = 2.0,
        init_imp: float = 0.0,
        learnable_imp: bool = True,
        detach_reset: bool = True
    ):
        super().__init__()
        
        self.num_features = num_features
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        
        # Membrane decay
        beta = 1.0 - 1.0 / tau
        self.register_buffer('beta', torch.tensor(beta))
        
        # Learnable initial membrane potential
        if learnable_imp:
            self.imp = nn.Parameter(torch.full((num_features,), init_imp))
        else:
            self.register_buffer('imp', torch.full((num_features,), init_imp))
        
        self.learnable_imp = learnable_imp
        self.v = None
        self.first_step = True
    
    def reset_state(self):
        self.v = None
        self.first_step = True
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with learnable IMP."""
        batch_size = x.size(0)
        
        # Initialize with learnable IMP on first step
        if self.v is None or self.first_step:
            imp = self.imp.view(1, -1).expand(batch_size, -1)
            if x.dim() == 4:  # Conv case
                imp = imp.view(batch_size, -1, 1, 1)
            self.v = imp.clone()
            self.first_step = False
        
        # Standard LIF dynamics
        self.v = self.beta * self.v + (1 - self.beta) * x
        
        # Spike generation
        spike = surrogate_function(self.v - self.v_threshold)
        
        # Reset
        spike_for_reset = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_for_reset * self.v_threshold
        else:
            self.v = (1 - spike_for_reset) * self.v + spike_for_reset * self.v_reset
        
        return spike, self.v


class HeterogeneousNeuronLayer(nn.Module):
    """
    Layer with heterogeneous neuron parameters.
    
    Based on findings that neural heterogeneity improves learning
    efficiency and acts as implicit regularization.
    
    Each neuron can have different:
    - Membrane time constant (tau)
    - Threshold
    - Reset voltage
    
    This captures diverse temporal dynamics within a single layer,
    similar to biological neural populations.
    """
    
    def __init__(
        self,
        num_neurons: int,
        tau_range: Tuple[float, float] = (1.5, 4.0),
        threshold_range: Tuple[float, float] = (0.8, 1.2),
        distribution: str = 'uniform',  # 'uniform', 'log', 'gaussian'
        learnable: bool = True,
        detach_reset: bool = True
    ):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.detach_reset = detach_reset
        
        # Initialize heterogeneous parameters
        if distribution == 'uniform':
            taus = torch.rand(num_neurons) * (tau_range[1] - tau_range[0]) + tau_range[0]
            thresholds = torch.rand(num_neurons) * (threshold_range[1] - threshold_range[0]) + threshold_range[0]
        elif distribution == 'log':
            taus = torch.exp(
                torch.rand(num_neurons) * (math.log(tau_range[1]) - math.log(tau_range[0])) + 
                math.log(tau_range[0])
            )
            thresholds = torch.exp(
                torch.rand(num_neurons) * (math.log(threshold_range[1]) - math.log(threshold_range[0])) + 
                math.log(threshold_range[0])
            )
        else:  # gaussian
            tau_mean = (tau_range[0] + tau_range[1]) / 2
            tau_std = (tau_range[1] - tau_range[0]) / 4
            taus = torch.clamp(
                torch.randn(num_neurons) * tau_std + tau_mean,
                tau_range[0], tau_range[1]
            )
            thresh_mean = (threshold_range[0] + threshold_range[1]) / 2
            thresh_std = (threshold_range[1] - threshold_range[0]) / 4
            thresholds = torch.clamp(
                torch.randn(num_neurons) * thresh_std + thresh_mean,
                threshold_range[0], threshold_range[1]
            )
        
        # Convert to decay factors
        betas = 1.0 - 1.0 / taus
        
        if learnable:
            # Store as learnable parameters with sigmoid transformation
            self.beta_raw = nn.Parameter(torch.log(betas / (1 - betas + 1e-8)))
            self.threshold_raw = nn.Parameter(thresholds)
        else:
            self.register_buffer('beta_raw', betas)
            self.register_buffer('threshold_raw', thresholds)
        
        self.learnable = learnable
        self.v = None
    
    @property
    def beta(self) -> torch.Tensor:
        if self.learnable:
            return torch.sigmoid(self.beta_raw)
        return self.beta_raw
    
    @property
    def threshold(self) -> torch.Tensor:
        if self.learnable:
            return F.softplus(self.threshold_raw)  # Ensure positive
        return self.threshold_raw
    
    def reset_state(self):
        self.v = None
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with heterogeneous dynamics."""
        if self.v is None:
            self.v = torch.zeros_like(x)
        
        # Reshape parameters for broadcasting
        beta = self.beta.view(1, -1)
        threshold = self.threshold.view(1, -1)
        
        if x.dim() == 4:  # Conv case
            beta = beta.view(1, -1, 1, 1)
            threshold = threshold.view(1, -1, 1, 1)
        
        # Per-neuron membrane dynamics
        self.v = beta * self.v + (1 - beta) * x
        
        # Per-neuron thresholding
        spike = surrogate_function(self.v - threshold)
        
        # Reset
        spike_for_reset = spike.detach() if self.detach_reset else spike
        self.v = self.v - spike_for_reset * threshold
        
        return spike, self.v


# Import F for softplus
import torch.nn.functional as F
