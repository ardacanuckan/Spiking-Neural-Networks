"""
Surrogate Gradient Functions for Spiking Neural Networks

The core challenge in training SNNs is that the spike function (Heaviside)
has zero gradient almost everywhere. Surrogate gradients replace the 
true gradient with a smooth approximation during backpropagation.

This module implements various surrogate gradient functions with 
different characteristics (sharpness, support, computational cost).
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional
import math


class SurrogateFunction(nn.Module, ABC):
    """
    Base class for surrogate gradient functions.
    
    Forward pass: Heaviside step function (binary spike)
    Backward pass: Smooth surrogate gradient
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def surrogate_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute surrogate gradient."""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: Heaviside, Backward: Surrogate gradient
        """
        return SurrogateSpike.apply(x, self)


class SurrogateSpike(torch.autograd.Function):
    """
    Custom autograd function for spike with surrogate gradient.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, surrogate_fn: SurrogateFunction) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.surrogate_fn = surrogate_fn
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        grad = ctx.surrogate_fn.surrogate_gradient(x)
        return grad * grad_output, None


class Sigmoid(SurrogateFunction):
    """
    Sigmoid surrogate gradient.
    
    g(x) = alpha * sigmoid(alpha * x) * (1 - sigmoid(alpha * x))
    
    Properties:
    - Smooth and differentiable everywhere
    - Support: entire real line
    - Sharpness controlled by alpha
    
    Args:
        alpha: Sharpness parameter (higher = closer to step function)
    """
    
    def __init__(self, alpha: float = 4.0):
        super().__init__()
        self.alpha = alpha
    
    def surrogate_gradient(self, x: torch.Tensor) -> torch.Tensor:
        sigmoid_x = torch.sigmoid(self.alpha * x)
        return self.alpha * sigmoid_x * (1 - sigmoid_x)


class FastSigmoid(SurrogateFunction):
    """
    Fast Sigmoid (algebraic sigmoid) surrogate gradient.
    
    g(x) = 1 / (alpha * |x| + 1)^2
    
    Properties:
    - Computationally cheaper than true sigmoid
    - Heavy tails (slow decay)
    - Good gradient flow for distant spikes
    
    Args:
        alpha: Sharpness parameter
    """
    
    def __init__(self, alpha: float = 25.0):
        super().__init__()
        self.alpha = alpha
    
    def surrogate_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (self.alpha * torch.abs(x) + 1.0) ** 2


class ATan(SurrogateFunction):
    """
    Arctangent surrogate gradient.
    
    g(x) = 1 / (pi * (1 + (pi * alpha * x)^2))
    
    Properties:
    - Normalized to integrate to 1
    - Medium decay rate
    - Good balance between locality and gradient flow
    
    Args:
        alpha: Sharpness parameter
    """
    
    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha
    
    def surrogate_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (math.pi * (1 + (math.pi * self.alpha * x) ** 2))


class PiecewiseLinear(SurrogateFunction):
    """
    Piecewise Linear surrogate gradient.
    
    g(x) = max(0, 1 - alpha * |x|)
    
    Properties:
    - Finite support ([-1/alpha, 1/alpha])
    - Computationally very efficient
    - Sharp cutoff may cause gradient issues
    
    Args:
        alpha: Width parameter (1/width of support)
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def surrogate_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(1 - self.alpha * torch.abs(x), min=0)


class Gaussian(SurrogateFunction):
    """
    Gaussian surrogate gradient.
    
    g(x) = exp(-alpha * x^2)
    
    Properties:
    - Very smooth
    - Fast decay (Gaussian tails)
    - Biologically motivated (tuning curves)
    
    Args:
        alpha: Sharpness parameter (1/(2*sigma^2))
    """
    
    def __init__(self, alpha: float = 4.0):
        super().__init__()
        self.alpha = alpha
    
    def surrogate_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.alpha * x ** 2)


class SuperSpike(SurrogateFunction):
    """
    SuperSpike surrogate gradient.
    
    g(x) = 1 / (1 + beta * |x|)^2
    
    From "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"
    
    Properties:
    - Good for online learning
    - Moderate decay
    - Proven convergence properties
    
    Args:
        beta: Sharpness parameter
    """
    
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta
    
    def surrogate_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1 + self.beta * torch.abs(x)) ** 2


class MultiGaussian(SurrogateFunction):
    """
    Multi-Gaussian surrogate gradient.
    
    Sum of multiple Gaussian functions at different offsets,
    providing a richer gradient landscape.
    
    Args:
        num_gaussians: Number of Gaussian components
        sigma: Width of each Gaussian
        spacing: Spacing between Gaussian centers
    """
    
    def __init__(
        self,
        num_gaussians: int = 3,
        sigma: float = 0.5,
        spacing: float = 0.5
    ):
        super().__init__()
        self.sigma = sigma
        
        # Gaussian centers (symmetric around 0)
        half_n = num_gaussians // 2
        centers = torch.linspace(-half_n * spacing, half_n * spacing, num_gaussians)
        self.register_buffer('centers', centers)
    
    def surrogate_gradient(self, x: torch.Tensor) -> torch.Tensor:
        # Compute distance to each center
        x_expanded = x.unsqueeze(-1)  # [..., 1]
        distances = (x_expanded - self.centers) ** 2  # [..., num_gaussians]
        
        # Sum of Gaussians
        gaussians = torch.exp(-distances / (2 * self.sigma ** 2))
        return gaussians.sum(dim=-1) / len(self.centers)


class AdaptiveSurrogate(SurrogateFunction):
    """
    Adaptive surrogate gradient with learnable sharpness.
    
    The sharpness parameter is learned during training,
    allowing the network to adapt gradient flow.
    
    Args:
        init_alpha: Initial sharpness
        base_function: Type of surrogate ('sigmoid', 'atan', 'gaussian')
    """
    
    def __init__(
        self,
        init_alpha: float = 4.0,
        base_function: str = 'sigmoid',
        alpha_min: float = 0.1,
        alpha_max: float = 50.0
    ):
        super().__init__()
        
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.base_function = base_function
        
        # Learnable alpha with bounds
        # Use softplus transformation to keep alpha positive and bounded
        init_raw = math.log(math.exp(init_alpha - alpha_min) - 1 + 1e-8)
        self.alpha_raw = nn.Parameter(torch.tensor(init_raw))
    
    @property
    def alpha(self) -> torch.Tensor:
        """Get bounded alpha value."""
        return self.alpha_min + torch.clamp(
            torch.nn.functional.softplus(self.alpha_raw),
            max=self.alpha_max - self.alpha_min
        )
    
    def surrogate_gradient(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        
        if self.base_function == 'sigmoid':
            sigmoid_x = torch.sigmoid(alpha * x)
            return alpha * sigmoid_x * (1 - sigmoid_x)
        elif self.base_function == 'atan':
            return 1.0 / (math.pi * (1 + (math.pi * alpha * x) ** 2))
        elif self.base_function == 'gaussian':
            return torch.exp(-alpha * x ** 2)
        else:  # fast_sigmoid
            return 1.0 / (alpha * torch.abs(x) + 1.0) ** 2


def get_surrogate_function(name: str, **kwargs) -> SurrogateFunction:
    """
    Factory function to get surrogate gradient by name.
    
    Args:
        name: Name of surrogate function
        **kwargs: Parameters for the surrogate function
        
    Returns:
        SurrogateFunction instance
    """
    surrogate_map = {
        'sigmoid': Sigmoid,
        'fast_sigmoid': FastSigmoid,
        'atan': ATan,
        'piecewise_linear': PiecewiseLinear,
        'gaussian': Gaussian,
        'superspike': SuperSpike,
        'multi_gaussian': MultiGaussian,
        'adaptive': AdaptiveSurrogate
    }
    
    if name.lower() not in surrogate_map:
        raise ValueError(f"Unknown surrogate function: {name}. "
                        f"Available: {list(surrogate_map.keys())}")
    
    return surrogate_map[name.lower()](**kwargs)
