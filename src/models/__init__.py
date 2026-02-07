"""
Neuron Models for Spiking Neural Networks

Contains implementations of:
- Leaky Integrate-and-Fire (LIF) neurons
- Dendritic LIF neurons with multi-compartment modeling
- Adaptive threshold neurons
"""

from .base import BaseNeuron
from .lif import LIFNeuron, ParametricLIF
from .dendritic import DendriticLIFNeuron, DendriticBranch
from .adaptive import AdaptiveLIFNeuron

__all__ = [
    'BaseNeuron',
    'LIFNeuron',
    'ParametricLIF', 
    'DendriticLIFNeuron',
    'DendriticBranch',
    'AdaptiveLIFNeuron'
]
