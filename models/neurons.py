"""
Spiking Neuron Models
=====================

Implements various spiking neuron types:
- LIFNeuron: Standard Leaky Integrate-and-Fire
- AdaptiveLIF: LIF with adaptive threshold
- TTFS_LIF: Time-To-First-Spike LIF (fires at most once)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .surrogate import spike_fn, atan_spike


class LIFNeuron(nn.Module):
    """
    Standard Leaky Integrate-and-Fire Neuron.
    
    Dynamics:
        v[t+1] = beta * v[t] + (1-beta) * I[t]
        s[t] = H(v[t] - threshold)
        v[t] = v[t] - s[t] * threshold  (soft reset)
    
    Args:
        tau: Membrane time constant (default: 2.0)
        v_threshold: Spike threshold (default: 1.0)
        v_reset: Reset potential (default: 0.0, soft reset)
        surrogate: Surrogate gradient method ('atan', 'superspike', etc.)
    """
    
    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate='atan'):
        super().__init__()
        
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate = surrogate
        
        # Decay factor
        self.beta = 1.0 - 1.0 / tau
        
        # Membrane potential state
        self.v_mem = None
    
    def reset_state(self):
        """Reset membrane potential."""
        self.v_mem = None
    
    def forward(self, x):
        """
        Args:
            x: Input current [batch, features]
        
        Returns:
            spike: Binary spike output [batch, features]
        """
        if self.v_mem is None:
            self.v_mem = torch.zeros_like(x)
        
        # Leaky integration
        self.v_mem = self.beta * self.v_mem + (1 - self.beta) * x
        
        # Spike generation with surrogate gradient
        spike = spike_fn(self.v_mem - self.v_threshold, method=self.surrogate)
        
        # Soft reset
        self.v_mem = self.v_mem - spike.detach() * self.v_threshold
        
        return spike


class AdaptiveLIF(nn.Module):
    """
    LIF Neuron with Adaptive Threshold.
    
    The threshold adapts based on recent activity:
        threshold[t] = base_threshold + offset + adaptation * activity
    
    Args:
        num_neurons: Number of neurons (for per-neuron parameters)
        tau: Membrane time constant
        base_threshold: Base spike threshold
        adaptation_rate: Rate of threshold adaptation
    """
    
    def __init__(self, num_neurons, tau=2.0, base_threshold=1.0, adaptation_rate=0.05):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.base_threshold = base_threshold
        
        # Learnable threshold offset per neuron
        self.threshold_offset = nn.Parameter(torch.zeros(num_neurons))
        
        # Decay factor
        beta = 1.0 - 1.0 / tau
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
        
        # Adaptation
        self.adaptation_rate = nn.Parameter(torch.tensor(adaptation_rate))
        self.register_buffer('running_activity', torch.zeros(num_neurons))
        
        # State
        self.v_mem = None
    
    def reset_state(self):
        self.v_mem = None
    
    def get_threshold(self):
        """Get current adaptive threshold."""
        threshold = self.base_threshold + self.threshold_offset
        if self.training:
            threshold = threshold + self.adaptation_rate.abs() * self.running_activity
        return threshold
    
    def forward(self, x):
        if self.v_mem is None:
            self.v_mem = torch.zeros_like(x)
        
        # Leaky integration
        self.v_mem = self.beta * self.v_mem + (1 - self.beta) * x
        
        # Adaptive threshold
        threshold = self.get_threshold()
        
        # Spike
        spike = spike_fn(self.v_mem - threshold)
        
        # Update running activity (EMA)
        if self.training:
            with torch.no_grad():
                batch_activity = spike.mean(dim=0)
                self.running_activity = 0.95 * self.running_activity + 0.05 * batch_activity
        
        # Soft reset
        self.v_mem = self.v_mem - spike.detach() * threshold * 0.9
        
        return spike


class TTFS_LIF(nn.Module):
    """
    Time-To-First-Spike LIF Neuron.
    
    Each neuron fires AT MOST once, implementing ultra-sparse coding.
    Based on Nature Communications 2024 paper.
    
    Args:
        num_neurons: Number of neurons
        tau: Membrane time constant
        base_threshold: Spike threshold
    """
    
    def __init__(self, num_neurons, tau=2.0, base_threshold=1.0):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.base_threshold = base_threshold
        
        # Learnable threshold offset
        self.threshold_offset = nn.Parameter(torch.zeros(num_neurons))
        
        # Decay
        beta = 1.0 - 1.0 / tau
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
        
        # State
        self.v_mem = None
        self.has_fired = None
        self.spike_times = None
    
    def reset_state(self, batch_size=None):
        self.v_mem = None
        self.has_fired = None
        self.spike_times = None
    
    def get_threshold(self):
        return self.base_threshold + F.softplus(self.threshold_offset)
    
    def forward(self, x, time_step=0, max_time=10):
        if self.v_mem is None:
            self.v_mem = torch.zeros_like(x)
            self.has_fired = torch.zeros_like(x, dtype=torch.bool)
            self.spike_times = torch.full_like(x, float(max_time))
        
        # Leaky integration
        self.v_mem = self.beta * self.v_mem + (1 - self.beta) * x
        
        # Threshold
        threshold = self.get_threshold()
        
        # Spike only if not already fired
        can_fire = ~self.has_fired
        spike = spike_fn(self.v_mem - threshold) * can_fire.float()
        
        # Record spike times
        new_spikes = (spike > 0.5) & can_fire
        self.spike_times = torch.where(
            new_spikes,
            torch.full_like(self.spike_times, float(time_step)),
            self.spike_times
        )
        
        # Update fired status
        self.has_fired = self.has_fired | new_spikes
        
        # Reset only fired neurons
        self.v_mem = self.v_mem * (1 - spike.detach())
        
        return spike


class ParametricLIF(nn.Module):
    """
    Parametric LIF with learnable tau.
    
    The time constant tau is learned per-neuron.
    """
    
    def __init__(self, num_neurons, tau_init=2.0, v_threshold=1.0):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.v_threshold = v_threshold
        
        # Learnable log(tau) for positivity
        self.log_tau = nn.Parameter(torch.full((num_neurons,), float(torch.log(torch.tensor(tau_init)))))
        
        self.v_mem = None
    
    def reset_state(self):
        self.v_mem = None
    
    def get_beta(self):
        tau = torch.exp(self.log_tau).clamp(1.1, 20.0)
        return 1.0 - 1.0 / tau
    
    def forward(self, x):
        beta = self.get_beta()
        
        if self.v_mem is None:
            self.v_mem = torch.zeros_like(x)
        
        # Leaky integration with learnable decay
        self.v_mem = beta * self.v_mem + (1 - beta) * x
        
        # Spike
        spike = spike_fn(self.v_mem - self.v_threshold)
        
        # Soft reset
        self.v_mem = self.v_mem - spike.detach() * self.v_threshold
        
        return spike
