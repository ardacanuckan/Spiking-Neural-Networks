"""
SNN Models Package
==================

Contains all spiking neural network architectures:
- BaselineLIF: Standard LIF network
- DASNN: Dendritic Attention SNN
- SpikingKAN: Spiking Kolmogorov-Arnold Network
- NEXUS: Neural EXpressive Unified Spiking network
- APEX: All-Performance EXtreme SNN
"""

from .baseline_lif import BaselineLIF
from .dasnn import DASNN
from .spiking_kan import SpikingKAN
from .nexus_snn import NEXUSSNN
from .apex_snn import APEXSNN
from .neurons import LIFNeuron, AdaptiveLIF, TTFS_LIF
from .surrogate import spike_fn, ATanSurrogate, SuperSpike

__all__ = [
    'BaselineLIF',
    'DASNN', 
    'SpikingKAN',
    'NEXUSSNN',
    'APEXSNN',
    'LIFNeuron',
    'AdaptiveLIF',
    'TTFS_LIF',
    'spike_fn',
    'ATanSurrogate',
    'SuperSpike',
]
