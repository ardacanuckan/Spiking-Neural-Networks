"""
Surrogate Gradient Functions
============================

Implements various surrogate gradient functions for training SNNs with backpropagation.
"""

import torch
import torch.nn as nn
import numpy as np


class ATanSurrogate(torch.autograd.Function):
    """
    ArcTan surrogate gradient.
    Standard in SpikingJelly framework.
    """
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / (2 * (1 + (np.pi/2 * alpha * x)**2))
        return grad * grad_output, None


class SuperSpike(torch.autograd.Function):
    """
    SuperSpike surrogate gradient.
    Known for training stability.
    """
    @staticmethod
    def forward(ctx, x, beta=10.0):
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = 1.0 / (1.0 + ctx.beta * torch.abs(x)) ** 2
        return grad * grad_output, None


class TriangularSurrogate(torch.autograd.Function):
    """
    Triangular surrogate gradient.
    Simple and effective.
    """
    @staticmethod
    def forward(ctx, x, width=1.0):
        ctx.save_for_backward(x)
        ctx.width = width
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = torch.clamp(1 - torch.abs(x) / ctx.width, min=0)
        return grad * grad_output, None


class SigmoidSurrogate(torch.autograd.Function):
    """
    Sigmoid surrogate gradient.
    Smooth approximation.
    """
    @staticmethod
    def forward(ctx, x, alpha=4.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sgax = torch.sigmoid(ctx.alpha * x)
        grad = ctx.alpha * sgax * (1 - sgax)
        return grad * grad_output, None


def spike_fn(x, method='atan', **kwargs):
    """
    Unified spike function with selectable surrogate gradient.
    
    Args:
        x: Input tensor (membrane - threshold)
        method: 'atan', 'superspike', 'triangular', 'sigmoid'
        **kwargs: Additional arguments for specific methods
    
    Returns:
        Binary spike tensor
    """
    if method == 'atan':
        alpha = kwargs.get('alpha', 2.0)
        return ATanSurrogate.apply(x, alpha)
    elif method == 'superspike':
        beta = kwargs.get('beta', 10.0)
        return SuperSpike.apply(x, beta)
    elif method == 'triangular':
        width = kwargs.get('width', 1.0)
        return TriangularSurrogate.apply(x, width)
    elif method == 'sigmoid':
        alpha = kwargs.get('alpha', 4.0)
        return SigmoidSurrogate.apply(x, alpha)
    else:
        # Default: ATan
        return ATanSurrogate.apply(x, 2.0)


# Convenience functions
def atan_spike(x, alpha=2.0):
    return ATanSurrogate.apply(x, alpha)

def superspike(x, beta=10.0):
    return SuperSpike.apply(x, beta)

def triangular_spike(x, width=1.0):
    return TriangularSurrogate.apply(x, width)
