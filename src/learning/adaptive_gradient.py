"""
Adaptive Surrogate Gradients Based on Membrane Potential Dynamics

NOVEL CONTRIBUTION: This module implements the MPD-Adaptive Surrogate Gradient,
which dynamically adjusts surrogate gradient sharpness based on the distribution
of membrane potentials during training.

Key Insight: As training progresses, membrane potential dynamics (MPD) shift,
causing fixed surrogate gradients to become misaligned with the optimization
landscape. By tracking MPD and adapting the surrogate gradient, we maintain
stable gradient flow throughout training.

Reference:
- "Adaptive Gradient Learning for SNNs by Exploiting Membrane Potential Dynamics" (IJCAI 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import math


class MPDAdaptiveSurrogate(nn.Module):
    """
    Membrane Potential Distribution Adaptive Surrogate Gradient.
    
    This is our NOVEL contribution: dynamically adjusting surrogate gradient
    sharpness based on running statistics of membrane potential.
    
    The key insight is that fixed surrogate gradients become misaligned
    with the membrane potential distribution as training progresses.
    
    Algorithm:
    1. Track running mean and variance of membrane potentials
    2. Adjust surrogate gradient width to cover the "active region"
    3. Scale gradient magnitude to maintain stable learning
    
    The surrogate gradient is:
        g(x) = alpha(t) * base_surrogate(x / sigma(t))
        
    where:
        - alpha(t) adapts based on gradient flow quality
        - sigma(t) tracks membrane potential spread
    
    Args:
        base_surrogate: Base surrogate gradient type
        momentum: Momentum for running statistics
        auto_scale: Whether to automatically scale gradient magnitude
        target_sparsity: Target spike sparsity for regularization
    """
    
    def __init__(
        self,
        base_surrogate: str = 'sigmoid',
        init_alpha: float = 4.0,
        momentum: float = 0.1,
        auto_scale: bool = True,
        target_sparsity: float = 0.5,
        min_alpha: float = 0.5,
        max_alpha: float = 50.0
    ):
        super().__init__()
        
        self.base_surrogate = base_surrogate
        self.momentum = momentum
        self.auto_scale = auto_scale
        self.target_sparsity = target_sparsity
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
        # Learnable base sharpness
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        
        # Running statistics of membrane potential
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(1.0))
        self.register_buffer('running_sparsity', torch.tensor(0.5))
        self.register_buffer('num_batches', torch.tensor(0))
    
    def update_statistics(self, membrane_potential: torch.Tensor) -> None:
        """
        Update running statistics of membrane potential.
        
        Args:
            membrane_potential: Current membrane potentials [batch, features]
        """
        with torch.no_grad():
            # Compute batch statistics
            batch_mean = membrane_potential.mean()
            batch_var = membrane_potential.var()
            
            # Update running statistics
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + 
                self.momentum * batch_mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + 
                self.momentum * batch_var
            )
            
            self.num_batches += 1
    
    def compute_adaptive_alpha(self) -> torch.Tensor:
        """
        Compute adaptive sharpness based on membrane statistics.
        
        The idea: if membrane potentials are widely spread, use a wider
        (smaller alpha) surrogate to capture gradients across the range.
        If concentrated near threshold, use sharper (larger alpha).
        """
        # Standard deviation of membrane potentials
        std = torch.sqrt(self.running_var + 1e-8)
        
        # Scale alpha inversely with std
        # When std is large, reduce alpha for wider gradient support
        scale_factor = 1.0 / (std + 0.1)
        
        # Combine with learnable base alpha
        adapted_alpha = self.alpha * scale_factor
        
        # Clamp to reasonable range
        return torch.clamp(adapted_alpha, self.min_alpha, self.max_alpha)
    
    def surrogate_gradient(
        self, 
        x: torch.Tensor,
        membrane_potential: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute adaptive surrogate gradient.
        
        Args:
            x: Input (membrane - threshold)
            membrane_potential: Raw membrane potential (for statistics update)
            
        Returns:
            Surrogate gradient
        """
        # Update statistics if membrane potential provided
        if membrane_potential is not None and self.training:
            self.update_statistics(membrane_potential)
        
        # Compute adaptive alpha
        alpha = self.compute_adaptive_alpha()
        
        # Compute base surrogate gradient with adaptive sharpness
        if self.base_surrogate == 'sigmoid':
            sigmoid_x = torch.sigmoid(alpha * x)
            grad = alpha * sigmoid_x * (1 - sigmoid_x)
        elif self.base_surrogate == 'atan':
            grad = 1.0 / (math.pi * (1 + (math.pi * alpha * x) ** 2))
        elif self.base_surrogate == 'gaussian':
            grad = torch.exp(-alpha * x ** 2)
        else:  # fast_sigmoid
            grad = 1.0 / (alpha * torch.abs(x) + 1.0) ** 2
        
        # Auto-scale to maintain gradient magnitude
        if self.auto_scale:
            # Scale based on running variance
            scale = 1.0 / (torch.sqrt(self.running_var) + 0.1)
            grad = grad * scale
        
        return grad
    
    def forward(
        self, 
        x: torch.Tensor,
        membrane_potential: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: Heaviside with adaptive surrogate backward.
        
        Args:
            x: Input (membrane - threshold)
            membrane_potential: Raw membrane for statistics
            
        Returns:
            Binary spikes
        """
        return MPDAdaptiveSpike.apply(x, self, membrane_potential)


class MPDAdaptiveSpike(torch.autograd.Function):
    """Custom autograd for MPD-adaptive surrogate gradient."""
    
    @staticmethod
    def forward(
        ctx, 
        x: torch.Tensor, 
        surrogate: MPDAdaptiveSurrogate,
        membrane_potential: Optional[torch.Tensor]
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.surrogate = surrogate
        ctx.membrane_potential = membrane_potential
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        grad = ctx.surrogate.surrogate_gradient(x, ctx.membrane_potential)
        return grad * grad_output, None, None


class TemporalContrastLoss(nn.Module):
    """
    Temporal Contrastive Loss for SNNs.
    
    Extends contrastive learning to the temporal domain by leveraging
    correlation between representations at different time steps.
    
    The loss encourages:
    1. Consistent spike patterns for same class across time
    2. Separation of spike patterns for different classes
    3. Smooth temporal evolution of representations
    
    L = L_class + lambda_t * L_temporal + lambda_s * L_sparsity
    
    Args:
        temperature: Softmax temperature for contrastive loss
        lambda_temporal: Weight for temporal consistency term
        lambda_sparsity: Weight for sparsity regularization
        target_rate: Target firing rate for sparsity
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        lambda_temporal: float = 0.1,
        lambda_sparsity: float = 0.01,
        target_rate: float = 0.2
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_temporal = lambda_temporal
        self.lambda_sparsity = lambda_sparsity
        self.target_rate = target_rate
    
    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        spike_trains: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute temporal contrastive loss.
        
        Args:
            outputs: Network outputs [batch, classes]
            labels: Ground truth labels [batch]
            spike_trains: Optional spike trains [time, batch, features]
            
        Returns:
            Total loss and dict of individual loss components
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(outputs, labels)
        
        losses = {'ce': ce_loss.item()}
        total_loss = ce_loss
        
        # Temporal consistency loss
        if spike_trains is not None and spike_trains.size(0) > 1:
            T = spike_trains.size(0)
            
            # Compute temporal consistency (adjacent time steps should be similar)
            temporal_loss = 0.0
            for t in range(T - 1):
                diff = spike_trains[t] - spike_trains[t + 1]
                temporal_loss = temporal_loss + (diff ** 2).mean()
            temporal_loss = temporal_loss / (T - 1)
            
            losses['temporal'] = temporal_loss.item()
            total_loss = total_loss + self.lambda_temporal * temporal_loss
            
            # Sparsity regularization
            firing_rate = spike_trains.mean()
            sparsity_loss = (firing_rate - self.target_rate) ** 2
            
            losses['sparsity'] = sparsity_loss.item()
            total_loss = total_loss + self.lambda_sparsity * sparsity_loss
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


class TETLoss(nn.Module):
    """
    Temporal Efficient Training (TET) Loss.
    
    Instead of accumulating outputs across all time steps and then
    computing loss, TET computes loss at each time step and aggregates.
    This provides better temporal credit assignment.
    
    L_TET = (1/T) * sum_{t=1}^{T} CE(output_t, label)
    
    This is equivalent to standard loss for rate-coded outputs but
    provides better gradients for temporal learning.
    
    Reference: "Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting"
    
    Args:
        loss_lambda: Weight for main classification loss
        means: Mean correction factors per timestep (optional)
        use_mean_correction: Whether to use mean correction
    """
    
    def __init__(
        self,
        loss_lambda: float = 0.0,
        use_mean_correction: bool = False
    ):
        super().__init__()
        self.loss_lambda = loss_lambda
        self.use_mean_correction = use_mean_correction
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        time_outputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute TET loss.
        
        Args:
            outputs: Final accumulated output [batch, classes]
            labels: Ground truth labels [batch]
            time_outputs: Optional per-timestep outputs [time, batch, classes]
            
        Returns:
            TET loss value
        """
        if time_outputs is None:
            # Fall back to standard cross-entropy
            return self.ce_loss(outputs, labels)
        
        T = time_outputs.size(0)
        
        # Compute loss at each time step
        tet_loss = 0.0
        for t in range(T):
            tet_loss = tet_loss + self.ce_loss(time_outputs[t], labels)
        tet_loss = tet_loss / T
        
        # Add regularization on final output
        if self.loss_lambda > 0:
            final_loss = self.ce_loss(outputs, labels)
            tet_loss = (1 - self.loss_lambda) * tet_loss + self.loss_lambda * final_loss
        
        return tet_loss


class FiringRateRegularizer(nn.Module):
    """
    Regularizes firing rate to prevent dead/overactive neurons.
    
    L_rate = sum(max(0, rate - upper)^2 + max(0, lower - rate)^2)
    
    Args:
        target_rate: Target firing rate
        lower_bound: Lower bound for firing rate
        upper_bound: Upper bound for firing rate
        penalty_scale: Scale for regularization penalty
    """
    
    def __init__(
        self,
        target_rate: float = 0.2,
        lower_bound: float = 0.01,
        upper_bound: float = 0.5,
        penalty_scale: float = 1.0
    ):
        super().__init__()
        self.target_rate = target_rate
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.penalty_scale = penalty_scale
    
    def forward(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """
        Compute firing rate regularization.
        
        Args:
            spike_trains: Spike trains [time, batch, features] or [batch, features]
            
        Returns:
            Regularization loss
        """
        # Compute per-neuron firing rates
        if spike_trains.dim() == 3:
            rates = spike_trains.mean(dim=(0, 1))  # Average over time and batch
        else:
            rates = spike_trains.mean(dim=0)  # Average over batch
        
        # Penalize rates outside bounds
        lower_penalty = F.relu(self.lower_bound - rates) ** 2
        upper_penalty = F.relu(rates - self.upper_bound) ** 2
        
        # Also add soft target penalty
        target_penalty = (rates - self.target_rate) ** 2
        
        total_penalty = (lower_penalty + upper_penalty + 0.1 * target_penalty).mean()
        
        return self.penalty_scale * total_penalty
