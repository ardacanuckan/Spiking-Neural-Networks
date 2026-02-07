"""
Leaky Integrate-and-Fire (LIF) Neuron Models

Implements standard LIF and Parametric LIF neurons with surrogate gradients.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
from .base import BaseNeuron


class LIFNeuron(BaseNeuron):
    """
    Standard Leaky Integrate-and-Fire (LIF) Neuron.
    
    The membrane potential dynamics follow:
        dV/dt = -(V - V_rest)/tau + I(t)/C
        
    Discretized as:
        V[t+1] = beta * V[t] + (1 - beta) * I[t]
        
    where beta = exp(-dt/tau) is the decay factor.
    
    Args:
        tau: Membrane time constant
        v_threshold: Firing threshold
        v_reset: Reset voltage (None for soft reset)
        beta: Decay factor (alternative to tau)
        learnable_beta: Whether beta is learnable
    """
    
    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        beta: Optional[float] = None,
        learnable_beta: bool = False,
        detach_reset: bool = True
    ):
        super().__init__(
            v_threshold=v_threshold,
            v_reset=v_reset,
            detach_reset=detach_reset
        )
        
        # Calculate beta from tau if not provided
        if beta is None:
            beta = 1.0 - 1.0 / tau
        
        if learnable_beta:
            # Use sigmoid to constrain beta to (0, 1)
            self.beta_raw = nn.Parameter(torch.tensor(self._inverse_sigmoid(beta)))
        else:
            self.register_buffer('beta_raw', torch.tensor(beta))
            self._learnable_beta = False
        
        self._learnable_beta = learnable_beta
    
    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        """Inverse sigmoid for parameter initialization."""
        return torch.log(torch.tensor(x / (1 - x + 1e-8))).item()
    
    @property
    def beta(self) -> torch.Tensor:
        """Get the decay factor."""
        if self._learnable_beta:
            return torch.sigmoid(self.beta_raw)
        return self.beta_raw
    
    def neuronal_charge(self, x: torch.Tensor) -> None:
        """
        Integrate input current into membrane potential.
        
        V[t+1] = beta * V[t] + (1 - beta) * I[t]
        """
        self.init_state(x)
        self.v = self.beta * self.v + (1 - self.beta) * x
    
    def forward(
        self, 
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LIF neuron.
        
        Args:
            x: Input current [batch, features] or [batch, channels, H, W]
            surrogate_function: Surrogate gradient function
            
        Returns:
            Tuple of (spikes, membrane_potential)
        """
        # Charge: integrate input
        self.neuronal_charge(x)
        
        # Fire: generate spikes
        spike = self.neuronal_fire(surrogate_function)
        
        # Reset: update membrane potential
        self.neuronal_reset(spike)
        
        return spike, self.v


class ParametricLIF(BaseNeuron):
    """
    Parametric Leaky Integrate-and-Fire (PLIF) Neuron.
    
    All neuron parameters (tau, threshold) are learnable per-neuron,
    enabling the network to learn optimal temporal dynamics.
    
    Reference: "Incorporating Learnable Membrane Time Constant to Enhance 
    Learning of Spiking Neural Networks" (ICCV 2021)
    
    Args:
        num_features: Number of neurons (for per-neuron parameters)
        init_tau: Initial time constant
        v_threshold: Initial firing threshold
        learnable_threshold: Whether threshold is learnable
    """
    
    def __init__(
        self,
        num_features: int,
        init_tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        learnable_threshold: bool = True,
        detach_reset: bool = True
    ):
        super().__init__(
            v_threshold=v_threshold,
            v_reset=v_reset,
            learnable_threshold=learnable_threshold,
            detach_reset=detach_reset
        )
        
        self.num_features = num_features
        
        # Per-neuron learnable time constants
        init_beta = 1.0 - 1.0 / init_tau
        init_w = torch.log(torch.tensor(init_beta / (1 - init_beta + 1e-8)))
        self.w = nn.Parameter(init_w.expand(num_features).clone())
    
    @property
    def beta(self) -> torch.Tensor:
        """Get per-neuron decay factors."""
        return torch.sigmoid(self.w)
    
    @property
    def tau(self) -> torch.Tensor:
        """Get per-neuron time constants."""
        return 1.0 / (1.0 - self.beta + 1e-8)
    
    def neuronal_charge(self, x: torch.Tensor) -> None:
        """
        Integrate with per-neuron time constants.
        """
        self.init_state(x)
        
        # Reshape beta for broadcasting
        beta = self.beta
        if x.dim() == 2:
            beta = beta.view(1, -1)
        elif x.dim() == 4:
            beta = beta.view(1, -1, 1, 1)
        
        self.v = beta * self.v + (1 - beta) * x
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with learnable dynamics."""
        self.neuronal_charge(x)
        spike = self.neuronal_fire(surrogate_function)
        self.neuronal_reset(spike)
        return spike, self.v


class LeakyIntegrator(nn.Module):
    """
    Non-spiking leaky integrator for readout layers.
    
    Useful for regression tasks or as a final layer that
    outputs continuous membrane potential.
    """
    
    def __init__(self, tau: float = 2.0, learnable: bool = False):
        super().__init__()
        beta = 1.0 - 1.0 / tau
        
        if learnable:
            self.beta_raw = nn.Parameter(
                torch.tensor(torch.log(torch.tensor(beta / (1 - beta + 1e-8))))
            )
        else:
            self.register_buffer('beta_raw', torch.tensor(beta))
        
        self.learnable = learnable
        self.v = None
    
    @property
    def beta(self) -> torch.Tensor:
        if self.learnable:
            return torch.sigmoid(self.beta_raw)
        return self.beta_raw
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate input without firing."""
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.beta * self.v + (1 - self.beta) * x
        return self.v
    
    def reset_state(self):
        self.v = None
