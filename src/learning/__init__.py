"""
Learning Rules and Surrogate Gradients for SNNs

Contains:
- Surrogate gradient functions for backpropagation
- Adaptive surrogate gradients based on membrane potential dynamics
- STDP and other bio-plausible learning rules
"""

from .surrogate import (
    SurrogateFunction,
    Sigmoid,
    FastSigmoid,
    ATan,
    PiecewiseLinear,
    Gaussian,
    SuperSpike,
    AdaptiveSurrogate
)

from .adaptive_gradient import (
    MPDAdaptiveSurrogate,
    TemporalContrastLoss,
    TETLoss
)

__all__ = [
    'SurrogateFunction',
    'Sigmoid',
    'FastSigmoid', 
    'ATan',
    'PiecewiseLinear',
    'Gaussian',
    'SuperSpike',
    'AdaptiveSurrogate',
    'MPDAdaptiveSurrogate',
    'TemporalContrastLoss',
    'TETLoss'
]
