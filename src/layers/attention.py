"""
Attention Mechanisms for Spiking Neural Networks

NOVEL CONTRIBUTION: Dendritic Self-Attention
Uses dendritic computation as a biologically plausible attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
import math


class DendriticSelfAttention(nn.Module):
    """
    NOVEL: Dendritic Self-Attention Layer
    
    Implements attention through dendritic gating mechanisms rather
    than explicit query-key-value computations.
    
    Key Innovations:
    1. O(n) complexity vs O(n²) for standard attention
    2. Event-driven: only active dendrites contribute
    3. Biologically plausible: mimics cortical attention
    4. Multi-timescale: different tau per attention head
    
    The mechanism:
    - Each dendritic branch computes a "relevance score" for its input
    - Scores are normalized via softmax (competition between branches)
    - Values are weighted by normalized scores
    - Sparse spike output encodes attended information
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads (dendritic branches)
        tau_range: Range of time constants for multi-scale attention
        temperature: Softmax temperature
        v_threshold: Firing threshold for output spikes
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        tau_range: Tuple[float, float] = (2.0, 16.0),
        temperature: float = 1.0,
        v_threshold: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.v_threshold = v_threshold
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Temperature for attention sharpness
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Create time constants for each head
        tau_min, tau_max = tau_range
        taus = torch.logspace(
            math.log10(tau_min), math.log10(tau_max), num_heads
        )
        betas = 1.0 - 1.0 / taus
        self.register_buffer('betas', betas)
        
        # Projections (simpler than QKV - dendrites learn relevance directly)
        self.relevance_proj = nn.Linear(dim, num_heads)  # Per-head relevance scores
        self.value_proj = nn.Linear(dim, dim)  # Values to attend to
        self.output_proj = nn.Linear(dim, dim)
        
        # Gating mechanism (dendritic modulation)
        self.gate = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # State variables
        self.v = None  # Membrane potential
        self.v_relevance = None  # Relevance accumulator per head
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.relevance_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.xavier_uniform_(self.gate.weight)
    
    def reset_state(self):
        self.v = None
        self.v_relevance = None
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dendritic attention.
        
        Args:
            x: Input tensor [batch, seq_len, dim] or [batch, dim]
            surrogate_function: Surrogate gradient function
            
        Returns:
            Tuple of (attended_spikes, membrane_potential)
        """
        is_sequential = x.dim() == 3
        if not is_sequential:
            x = x.unsqueeze(1)
        
        B, N, C = x.shape
        
        # Initialize states
        if self.v is None:
            self.v = torch.zeros(B, N, C, device=x.device)
        if self.v_relevance is None:
            self.v_relevance = torch.zeros(B, N, self.num_heads, device=x.device)
        
        # Compute relevance scores per head
        relevance = self.relevance_proj(x)  # [B, N, num_heads]
        
        # Leaky integration of relevance (temporal accumulation)
        # Each head has different tau, capturing different timescales
        betas = self.betas.view(1, 1, -1)
        self.v_relevance = betas * self.v_relevance + (1 - betas) * relevance
        
        # Softmax over heads (competition for attention)
        attention = F.softmax(self.v_relevance / self.temperature, dim=-1)  # [B, N, H]
        
        # Compute values and reshape for multi-head
        values = self.value_proj(x)  # [B, N, C]
        values = values.view(B, N, self.num_heads, self.head_dim)  # [B, N, H, D]
        
        # Apply attention (weight values by attention scores)
        attended = (attention.unsqueeze(-1) * values).sum(dim=2)  # [B, N, D*H] via broadcast
        attended = attended.view(B, N, C)  # [B, N, C]
        
        # Dendritic gating (modulation)
        gate = torch.sigmoid(self.gate(x))
        gated = attended * gate
        
        # Output projection
        output = self.output_proj(gated)
        output = self.dropout(output)
        
        # Leaky integration at soma
        beta_avg = self.betas.mean()
        self.v = beta_avg * self.v + (1 - beta_avg) * output
        
        # Spike generation
        spike = surrogate_function(self.v - self.v_threshold)
        
        # Reset
        spike_detach = spike.detach()
        self.v = self.v - spike_detach * self.v_threshold
        
        if not is_sequential:
            spike = spike.squeeze(1)
            self.v = self.v.squeeze(1)
        
        return spike, self.v


class SpikingMultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention adapted for Spiking Neural Networks.
    
    Uses spike-based Q, K, V computation with spiking output.
    This follows the Spikformer/Spikingformer architecture.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        tau: Membrane time constant
        v_threshold: Firing threshold
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.v_threshold = v_threshold
        
        beta = 1.0 - 1.0 / tau
        self.register_buffer('beta', torch.tensor(beta))
        
        # QKV projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Membrane potentials for Q, K, V spiking
        self.v_q = None
        self.v_k = None
        self.v_v = None
        self.v_out = None
    
    def reset_state(self):
        self.v_q = None
        self.v_k = None
        self.v_v = None
        self.v_out = None
    
    def forward(
        self,
        x: torch.Tensor,
        surrogate_function: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with spiking attention.
        
        Uses spike-based attention computation following:
        Attention = Softmax(Q @ K^T / sqrt(d)) @ V
        
        but with spiking Q, K, V.
        """
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Initialize membrane potentials
        if self.v_q is None:
            self.v_q = torch.zeros_like(q)
            self.v_k = torch.zeros_like(k)
            self.v_v = torch.zeros_like(v)
        
        # Leaky integration
        self.v_q = self.beta * self.v_q + (1 - self.beta) * q
        self.v_k = self.beta * self.v_k + (1 - self.beta) * k
        self.v_v = self.beta * self.v_v + (1 - self.beta) * v
        
        # Generate spikes for Q, K, V
        q_spike = surrogate_function(self.v_q - self.v_threshold)
        k_spike = surrogate_function(self.v_k - self.v_threshold)
        v_spike = surrogate_function(self.v_v - self.v_threshold)
        
        # Reset
        self.v_q = self.v_q - q_spike.detach() * self.v_threshold
        self.v_k = self.v_k - k_spike.detach() * self.v_threshold
        self.v_v = self.v_v - v_spike.detach() * self.v_threshold
        
        # Attention computation with spikes
        # Using spike counts as attention (addition-only, energy efficient)
        attn = (q_spike @ k_spike.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to spiked values
        out = (attn @ v_spike).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Output spiking
        if self.v_out is None:
            self.v_out = torch.zeros_like(out)
        
        self.v_out = self.beta * self.v_out + (1 - self.beta) * out
        out_spike = surrogate_function(self.v_out - self.v_threshold)
        self.v_out = self.v_out - out_spike.detach() * self.v_threshold
        
        return out_spike, self.v_out
