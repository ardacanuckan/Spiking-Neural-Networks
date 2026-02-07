"""
Base Neuron Model for Spiking Neural Networks

Provides the abstract interface for all neuron models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any


class BaseNeuron(nn.Module, ABC):
    """
    Abstract base class for spiking neuron models.
    
    All neuron models should inherit from this class and implement
    the required methods for membrane dynamics and spike generation.
    
    Attributes:
        v_threshold: Firing threshold voltage
        v_reset: Reset voltage after spike (None for soft reset)
        v_rest: Resting membrane potential
        detach_reset: Whether to detach reset from computational graph
    """
    
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        v_rest: float = 0.0,
        detach_reset: bool = True,
        learnable_threshold: bool = False
    ):
        super().__init__()
        self.v_rest = v_rest
        self.detach_reset = detach_reset
        
        # Learnable or fixed threshold
        if learnable_threshold:
            self.v_threshold = nn.Parameter(torch.tensor(v_threshold))
        else:
            self.register_buffer('v_threshold', torch.tensor(v_threshold))
        
        # Reset voltage (None = soft reset)
        if v_reset is not None:
            self.register_buffer('v_reset', torch.tensor(v_reset))
        else:
            self.v_reset = None
        
        # State variables
        self.v = None  # Membrane potential
        
    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor) -> None:
        """
        Update membrane potential based on input current.
        
        Args:
            x: Input current tensor
        """
        pass
    
    def neuronal_fire(self, surrogate_function) -> torch.Tensor:
        """
        Generate spikes based on membrane potential.
        
        Args:
            surrogate_function: Surrogate gradient function for backprop
            
        Returns:
            Binary spike tensor
        """
        return surrogate_function(self.v - self.v_threshold)
    
    def neuronal_reset(self, spike: torch.Tensor) -> None:
        """
        Reset membrane potential after spike.
        
        Args:
            spike: Binary spike tensor
        """
        if self.detach_reset:
            spike = spike.detach()
            
        if self.v_reset is None:
            # Soft reset: subtract threshold
            self.v = self.v - spike * self.v_threshold
        else:
            # Hard reset: set to reset voltage
            self.v = (1 - spike) * self.v + spike * self.v_reset
    
    def reset_state(self, batch_size: Optional[int] = None, 
                    device: Optional[torch.device] = None) -> None:
        """
        Reset the neuron state (membrane potential).
        
        Args:
            batch_size: Batch size for state initialization
            device: Device for state tensors
        """
        self.v = None
        
    def init_state(self, x: torch.Tensor) -> None:
        """
        Initialize membrane potential based on input shape.
        
        Args:
            x: Input tensor to determine shape
        """
        if self.v is None:
            self.v = torch.zeros_like(x)
    
    @abstractmethod
    def forward(self, x: torch.Tensor, 
                surrogate_function) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: charge -> fire -> reset
        
        Args:
            x: Input current
            surrogate_function: Surrogate gradient function
            
        Returns:
            Tuple of (spikes, membrane_potential)
        """
        pass
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current neuron state."""
        return {'v': self.v}
    
    def set_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Set neuron state."""
        self.v = state.get('v', self.v)


class NeuronState:
    """
    Container for neuron state across time steps.
    
    Useful for tracking membrane potential evolution and spike patterns.
    """
    
    def __init__(self):
        self.membrane_potentials = []
        self.spikes = []
        self.threshold_history = []
        
    def record(self, v: torch.Tensor, spike: torch.Tensor, 
               threshold: Optional[torch.Tensor] = None):
        """Record state at current time step."""
        self.membrane_potentials.append(v.detach().cpu())
        self.spikes.append(spike.detach().cpu())
        if threshold is not None:
            self.threshold_history.append(threshold.detach().cpu())
    
    def get_firing_rate(self) -> torch.Tensor:
        """Calculate average firing rate."""
        if not self.spikes:
            return torch.tensor(0.0)
        return torch.stack(self.spikes).mean(dim=0)
    
    def get_spike_times(self) -> torch.Tensor:
        """Get indices of time steps where spikes occurred."""
        if not self.spikes:
            return torch.tensor([])
        spikes_tensor = torch.stack(self.spikes)
        return torch.nonzero(spikes_tensor)
    
    def clear(self):
        """Clear recorded history."""
        self.membrane_potentials = []
        self.spikes = []
        self.threshold_history = []
