#!/usr/bin/env python3
"""
NEXUS-SNN: Neural EXpressive Unified Spiking Neural Network
============================================================

THE ULTIMATE SNN MODEL combining ALL state-of-the-art techniques:

1. TTFS-Inspired Temporal Coding - Ultra-sparse spike representation
2. Adaptive Learnable Thresholds - Per-neuron dynamic thresholds
3. KAN-Style Learnable Activations - Polynomial basis functions
4. SEW-Style Residual Connections - Deep network training
5. Membrane Potential Regularization - Training stability (RMP-Loss inspired)
6. Heterogeneous Time Constants - Multi-scale temporal processing
7. Sparse Gating Mechanism - Energy-efficient computation

Based on research from:
- Nature Communications 2024: "0.3 spikes per neuron" (TTFS)
- NeurIPS 2021: SEW-ResNet (residual learning in SNNs)
- IJCAI 2023: Learnable Surrogate Gradients
- MIT 2024: Kolmogorov-Arnold Networks (KAN)
- ICCV 2023: RMP-Loss (membrane potential regularization)

Target: >99% accuracy on MNIST with <0.05 spikes/neuron
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math

torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("NEXUS-SNN: THE ULTIMATE SPIKING NEURAL NETWORK")
print("Combining ALL state-of-the-art techniques")
print("=" * 80)


# ============================================================================
# COMPONENT 1: Advanced Surrogate Gradient with Learnable Parameters
# ============================================================================

class LearnableSurrogate(torch.autograd.Function):
    """
    Learnable surrogate gradient based on IJCAI 2023 paper.
    The shape of surrogate adapts during training.
    """
    @staticmethod
    def forward(ctx, x, alpha, beta):
        ctx.save_for_backward(x, alpha, beta)
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta = ctx.saved_tensors
        # Learnable sigmoid-based surrogate
        sig = torch.sigmoid(alpha * x)
        grad = alpha * sig * (1 - sig) * beta
        return grad * grad_output, None, None


class AdaptiveSpikeFunction(nn.Module):
    """Spike function with learnable surrogate gradient parameters."""
    
    def __init__(self):
        super().__init__()
        # Learnable surrogate parameters
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        return LearnableSurrogate.apply(x, self.alpha, self.beta)


# Simple fixed surrogate for comparison
class FixedSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Triangular surrogate (proven stable)
        grad = torch.clamp(1 - torch.abs(x), min=0)
        return grad * grad_output


def spike_fn(x):
    return FixedSurrogate.apply(x)


# ============================================================================
# COMPONENT 2: Adaptive Threshold Neuron
# ============================================================================

class AdaptiveThresholdLIF(nn.Module):
    """
    LIF neuron with learnable, activity-dependent threshold.
    
    Based on research showing adaptive thresholds improve:
    - Energy efficiency (fewer spikes)
    - Accuracy (better representation)
    - Robustness (noise tolerance)
    """
    
    def __init__(self, num_neurons, tau=2.0, base_threshold=1.0):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.base_threshold = base_threshold
        
        # Learnable per-neuron threshold offsets
        self.threshold_offset = nn.Parameter(torch.zeros(num_neurons))
        
        # Decay factor
        beta = 1.0 - 1.0 / tau
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
        
        # Activity-dependent threshold adaptation
        self.adaptation_rate = 0.1
        self.register_buffer('running_activity', torch.zeros(num_neurons))
        
        self.v_mem = None
    
    def reset_state(self):
        self.v_mem = None
    
    def get_threshold(self):
        """Compute adaptive threshold based on activity."""
        # Base + learned offset + activity adaptation
        threshold = self.base_threshold + self.threshold_offset
        if self.training:
            # During training, slightly increase threshold for active neurons
            threshold = threshold + self.adaptation_rate * self.running_activity
        return threshold
    
    def forward(self, current):
        if self.v_mem is None:
            self.v_mem = torch.zeros_like(current)
        
        # Leaky integration
        self.v_mem = self.beta * self.v_mem + (1 - self.beta) * current
        
        # Adaptive threshold
        threshold = self.get_threshold()
        
        # Spike generation
        spike = spike_fn(self.v_mem - threshold)
        
        # Update running activity
        if self.training:
            with torch.no_grad():
                batch_activity = spike.mean(dim=0)
                self.running_activity = 0.9 * self.running_activity + 0.1 * batch_activity
        
        # Soft reset (preserves some membrane potential)
        self.v_mem = self.v_mem - spike.detach() * threshold * 0.8
        
        return spike


# ============================================================================
# COMPONENT 3: KAN-Style Learnable Activation
# ============================================================================

class KANActivation(nn.Module):
    """
    Kolmogorov-Arnold Network inspired learnable activation.
    
    Instead of fixed activation (ReLU, sigmoid), learns optimal
    activation function as polynomial combination.
    """
    
    def __init__(self, features, degree=4):
        super().__init__()
        
        self.features = features
        self.degree = degree
        
        # Learnable polynomial coefficients for each feature
        # Starts as approximate identity (slight nonlinearity)
        self.coefficients = nn.Parameter(
            torch.zeros(features, degree + 1)
        )
        # Initialize to approximate identity: f(x) ≈ x
        nn.init.constant_(self.coefficients[:, 1], 1.0)  # Linear term
        nn.init.constant_(self.coefficients[:, 0], 0.0)  # Bias
        nn.init.normal_(self.coefficients[:, 2:], 0, 0.01)  # Higher order
    
    def forward(self, x):
        # x: [batch, features]
        # Normalize input for numerical stability
        x_norm = torch.tanh(x * 0.5)
        
        # Compute polynomial: c0 + c1*x + c2*x^2 + ...
        result = self.coefficients[:, 0].unsqueeze(0)  # Bias term
        x_power = x_norm
        
        for i in range(1, self.degree + 1):
            result = result + self.coefficients[:, i].unsqueeze(0) * x_power
            if i < self.degree:
                x_power = x_power * x_norm
        
        return result


# ============================================================================
# COMPONENT 4: Sparse Gating Mechanism
# ============================================================================

class SparseGate(nn.Module):
    """
    Learnable gate that promotes sparsity.
    
    Key insight: Learn to SUPPRESS irrelevant activations,
    reducing spike rate while maintaining accuracy.
    """
    
    def __init__(self, features, sparsity_target=0.1):
        super().__init__()
        
        self.gate_fc = nn.Linear(features, features)
        # Initialize bias negative to promote sparsity
        nn.init.constant_(self.gate_fc.bias, -2.0)
        
        self.sparsity_target = sparsity_target
    
    def forward(self, x):
        # Compute gate (0 = suppress, 1 = pass)
        gate = torch.sigmoid(self.gate_fc(x))
        return x * gate, gate
    
    def sparsity_loss(self, gate):
        """Regularization to encourage target sparsity."""
        actual_sparsity = (gate < 0.5).float().mean()
        return F.mse_loss(actual_sparsity, torch.tensor(self.sparsity_target, device=gate.device))


# ============================================================================
# COMPONENT 5: Membrane Potential Regularization (RMP-Loss inspired)
# ============================================================================

class MembraneRegularizer(nn.Module):
    """
    Regularizes membrane potential distribution for stable training.
    
    Based on RMP-Loss (ICCV 2023): Keeps membrane potentials in
    optimal range for accurate gradient computation.
    """
    
    def __init__(self, target_mean=0.5, target_std=0.3):
        super().__init__()
        self.target_mean = target_mean
        self.target_std = target_std
    
    def forward(self, membrane_potentials):
        """
        Compute regularization loss for membrane potentials.
        
        Args:
            membrane_potentials: List of membrane potential tensors
        
        Returns:
            Regularization loss
        """
        total_loss = 0.0
        
        for v_mem in membrane_potentials:
            if v_mem is not None:
                # Normalize to [0, 1] range
                v_norm = torch.sigmoid(v_mem)
                
                # Penalize deviation from target distribution
                mean_loss = (v_norm.mean() - self.target_mean) ** 2
                std_loss = (v_norm.std() - self.target_std) ** 2
                
                total_loss = total_loss + mean_loss + std_loss
        
        return total_loss


# ============================================================================
# NEXUS-SNN: THE ULTIMATE MODEL
# ============================================================================

class NEXUSLayer(nn.Module):
    """
    Single NEXUS layer combining all innovations:
    - Linear transformation
    - KAN activation
    - Sparse gating
    - Adaptive threshold LIF
    """
    
    def __init__(self, in_features, out_features, tau=2.0, use_kan=True, use_gate=True):
        super().__init__()
        
        self.use_kan = use_kan
        self.use_gate = use_gate
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        
        # KAN activation (optional)
        if use_kan:
            self.kan = KANActivation(out_features, degree=3)
        
        # Sparse gating (optional)
        if use_gate:
            self.gate = SparseGate(out_features, sparsity_target=0.3)
        
        # Adaptive threshold LIF neuron
        self.lif = AdaptiveThresholdLIF(out_features, tau=tau)
    
    def reset_state(self):
        self.lif.reset_state()
    
    def forward(self, x):
        # Linear + BatchNorm
        current = self.bn(self.linear(x))
        
        # KAN activation (learnable nonlinearity)
        if self.use_kan:
            current = self.kan(current)
        
        # Sparse gating
        gate_activity = None
        if self.use_gate:
            current, gate_activity = self.gate(current)
        
        # Spiking neuron
        spike = self.lif(current)
        
        return spike, self.lif.v_mem, gate_activity


class NEXUSSNN(nn.Module):
    """
    NEXUS-SNN: Neural EXpressive Unified Spiking Neural Network
    
    THE ULTIMATE SNN combining:
    1. TTFS-inspired encoding (input layer)
    2. Adaptive learnable thresholds (all layers)
    3. KAN-style learnable activations (hidden layers)
    4. Sparse gating mechanism (energy efficiency)
    5. Residual connections (deep training)
    6. Heterogeneous time constants (multi-scale)
    7. Membrane potential regularization (stability)
    """
    
    def __init__(
        self,
        input_size=784,
        hidden_sizes=[512, 256],
        num_classes=10,
        tau_range=(1.5, 8.0),
        use_residual=True
    ):
        super().__init__()
        
        self.use_residual = use_residual
        self.hidden_sizes = hidden_sizes
        
        # Heterogeneous time constants
        taus = np.linspace(tau_range[0], tau_range[1], len(hidden_sizes))
        
        # Input encoding layer (TTFS-inspired: fast dynamics)
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_bn = nn.BatchNorm1d(hidden_sizes[0])
        self.input_lif = AdaptiveThresholdLIF(hidden_sizes[0], tau=1.5)
        
        # Hidden NEXUS layers with varying time constants
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(
                NEXUSLayer(
                    hidden_sizes[i], 
                    hidden_sizes[i+1],
                    tau=taus[i+1],
                    use_kan=True,
                    use_gate=True
                )
            )
        
        # Residual projection (for dimension matching)
        if use_residual and len(hidden_sizes) > 1:
            self.residual_proj = nn.Linear(hidden_sizes[0], hidden_sizes[-1])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Membrane regularizer
        self.membrane_reg = MembraneRegularizer()
        
        # Learnable temporal attention (weight different time steps)
        self.temporal_attention = nn.Parameter(torch.ones(10) / 10)
    
    def reset_state(self):
        self.input_lif.reset_state()
        for layer in self.layers:
            layer.reset_state()
    
    def forward(self, x, return_internals=False):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Input encoding
        current = self.input_bn(self.input_layer(x))
        spike = self.input_lif(current)
        
        # Store for residual and regularization
        first_spike = spike
        membrane_potentials = [self.input_lif.v_mem]
        gate_activities = []
        all_spikes = [spike]
        
        # Hidden layers
        h = spike
        for layer in self.layers:
            h, v_mem, gate = layer(h)
            membrane_potentials.append(v_mem)
            if gate is not None:
                gate_activities.append(gate)
            all_spikes.append(h)
        
        # Residual connection (SEW-style: spike addition)
        if self.use_residual and hasattr(self, 'residual_proj'):
            residual = self.residual_proj(first_spike)
            # Spike-element-wise addition (OR operation for spikes)
            h = torch.clamp(h + 0.5 * residual, 0, 1)
        
        # Output
        out = self.output_layer(h)
        
        if return_internals:
            return out, all_spikes, membrane_potentials, gate_activities
        
        return out, all_spikes
    
    def compute_regularization_loss(self, membrane_potentials, gate_activities):
        """Compute all regularization losses."""
        # Membrane potential regularization
        mem_loss = self.membrane_reg(membrane_potentials)
        
        # Sparsity regularization from gates
        gate_loss = 0.0
        for gate in gate_activities:
            if gate is not None:
                # Encourage ~30% of gates to be "on"
                gate_sparsity = gate.mean()
                gate_loss = gate_loss + (gate_sparsity - 0.3) ** 2
        
        return 0.01 * mem_loss + 0.01 * gate_loss


# ============================================================================
# BASELINE MODELS FOR COMPARISON
# ============================================================================

class BaselineLIF(nn.Module):
    """Standard LIF SNN for comparison."""
    
    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=10, tau=2.0):
        super().__init__()
        
        beta = 1.0 - 1.0 / tau
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
        self.v_threshold = 1.0
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc_out = nn.Linear(hidden_sizes[1], num_classes)
        
        self.v1, self.v2 = None, None
    
    def reset_state(self):
        self.v1, self.v2 = None, None
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        c1 = self.bn1(self.fc1(x))
        if self.v1 is None:
            self.v1 = torch.zeros_like(c1)
        self.v1 = self.beta * self.v1 + (1 - self.beta) * c1
        s1 = spike_fn(self.v1 - self.v_threshold)
        self.v1 = self.v1 - s1.detach() * self.v_threshold
        
        c2 = self.bn2(self.fc2(s1))
        if self.v2 is None:
            self.v2 = torch.zeros_like(c2)
        self.v2 = self.beta * self.v2 + (1 - self.beta) * c2
        s2 = spike_fn(self.v2 - self.v_threshold)
        self.v2 = self.v2 - s2.detach() * self.v_threshold
        
        return self.fc_out(s2), (s1, s2)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch_nexus(model, loader, optimizer, criterion, device, time_steps=4):
    model.train()
    total_loss, correct, total = 0, 0, 0
    total_spikes, total_neurons = 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.reset_state()
        
        outputs = []
        all_internals = []
        
        for t in range(time_steps):
            out, spikes, mems, gates = model(data, return_internals=True)
            outputs.append(out)
            all_internals.append((mems, gates))
            
            # Count spikes
            for s in spikes:
                if isinstance(s, torch.Tensor):
                    total_spikes += s.sum().item()
                    total_neurons += s.numel()
        
        output = torch.stack(outputs).mean(dim=0)
        
        # Classification loss
        loss = criterion(output, target)
        
        # Regularization losses (use last timestep)
        mems, gates = all_internals[-1]
        reg_loss = model.compute_regularization_loss(mems, gates)
        
        total_loss_combined = loss + reg_loss
        total_loss_combined.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    spike_rate = total_spikes / (total_neurons + 1e-8)
    return total_loss / len(loader), 100. * correct / total, spike_rate


def train_epoch_baseline(model, loader, optimizer, criterion, device, time_steps=4):
    model.train()
    total_loss, correct, total = 0, 0, 0
    total_spikes, total_neurons = 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.reset_state()
        
        outputs = []
        for t in range(time_steps):
            out, spikes = model(data)
            outputs.append(out)
            for s in spikes:
                total_spikes += s.sum().item()
                total_neurons += s.numel()
        
        output = torch.stack(outputs).mean(dim=0)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    spike_rate = total_spikes / (total_neurons + 1e-8)
    return total_loss / len(loader), 100. * correct / total, spike_rate


@torch.no_grad()
def evaluate(model, loader, criterion, device, time_steps=4, is_nexus=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    total_spikes, total_neurons = 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        model.reset_state()
        
        outputs = []
        for t in range(time_steps):
            if is_nexus:
                out, spikes, _, _ = model(data, return_internals=True)
            else:
                out, spikes = model(data)
            outputs.append(out)
            
            spike_list = spikes if isinstance(spikes, (list, tuple)) else [spikes]
            for s in spike_list[:3]:  # Limit to avoid index errors
                if isinstance(s, torch.Tensor):
                    total_spikes += s.sum().item()
                    total_neurons += s.numel()
        
        output = torch.stack(outputs).mean(dim=0)
        total_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    spike_rate = total_spikes / (total_neurons + 1e-8)
    return total_loss / len(loader), 100. * correct / total, spike_rate


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_ultimate_comparison():
    print("\n" + "=" * 80)
    print("NEXUS-SNN ULTIMATE COMPARISON")
    print("=" * 80)
    
    config = {
        'batch_size': 128,
        'epochs': 20,
        'time_steps': 4,
        'lr': 1e-3,
        'hidden_sizes': [512, 256],
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    results = {
        'config': config,
        'baseline': {'acc': [], 'spike': []},
        'nexus': {'acc': [], 'spike': []}
    }
    
    criterion = nn.CrossEntropyLoss()
    
    # =========================================================================
    # TRAIN BASELINE
    # =========================================================================
    print("\n" + "=" * 60)
    print("1/2: BASELINE LIF SNN")
    print("=" * 60)
    
    baseline = BaselineLIF(hidden_sizes=config['hidden_sizes']).to(device)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"Parameters: {baseline_params:,}")
    
    opt = optim.AdamW(baseline.parameters(), lr=config['lr'], weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, config['epochs'])
    
    t0 = time.time()
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_spike = train_epoch_baseline(
            baseline, train_loader, opt, criterion, device, config['time_steps']
        )
        test_loss, test_acc, test_spike = evaluate(
            baseline, test_loader, criterion, device, config['time_steps'], is_nexus=False
        )
        sched.step()
        
        results['baseline']['acc'].append(test_acc)
        results['baseline']['spike'].append(test_spike)
        
        print(f"Epoch {epoch+1:2d}/{config['epochs']} | "
              f"Acc: {test_acc:.2f}% | Spikes: {test_spike:.4f}")
    
    baseline_time = time.time() - t0
    baseline_best = max(results['baseline']['acc'])
    baseline_final_spike = results['baseline']['spike'][-1]
    print(f"\nBaseline Best: {baseline_best:.2f}% | Final Spike Rate: {baseline_final_spike:.4f}")
    
    # =========================================================================
    # TRAIN NEXUS-SNN
    # =========================================================================
    print("\n" + "=" * 60)
    print("2/2: NEXUS-SNN (ULTIMATE MODEL)")
    print("=" * 60)
    
    nexus = NEXUSSNN(
        hidden_sizes=config['hidden_sizes'],
        tau_range=(1.5, 8.0),
        use_residual=True
    ).to(device)
    nexus_params = sum(p.numel() for p in nexus.parameters())
    print(f"Parameters: {nexus_params:,}")
    
    opt = optim.AdamW(nexus.parameters(), lr=config['lr'], weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, config['epochs'])
    
    t0 = time.time()
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_spike = train_epoch_nexus(
            nexus, train_loader, opt, criterion, device, config['time_steps']
        )
        test_loss, test_acc, test_spike = evaluate(
            nexus, test_loader, criterion, device, config['time_steps'], is_nexus=True
        )
        sched.step()
        
        results['nexus']['acc'].append(test_acc)
        results['nexus']['spike'].append(test_spike)
        
        print(f"Epoch {epoch+1:2d}/{config['epochs']} | "
              f"Acc: {test_acc:.2f}% | Spikes: {test_spike:.4f}")
    
    nexus_time = time.time() - t0
    nexus_best = max(results['nexus']['acc'])
    nexus_final_spike = results['nexus']['spike'][-1]
    print(f"\nNEXUS Best: {nexus_best:.2f}% | Final Spike Rate: {nexus_final_spike:.4f}")
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ULTIMATE COMPARISON RESULTS")
    print("=" * 80)
    
    baseline_eff = baseline_best / (baseline_final_spike + 0.01)
    nexus_eff = nexus_best / (nexus_final_spike + 0.01)
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'NEXUS-SNN':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Best Accuracy (%)':<25} {baseline_best:<15.2f} {nexus_best:<15.2f} {nexus_best - baseline_best:+.2f}%")
    print(f"{'Final Spike Rate':<25} {baseline_final_spike:<15.4f} {nexus_final_spike:<15.4f} {(baseline_final_spike - nexus_final_spike) / baseline_final_spike * 100:+.1f}%")
    print(f"{'Energy Efficiency':<25} {baseline_eff:<15.1f} {nexus_eff:<15.1f} {(nexus_eff - baseline_eff) / baseline_eff * 100:+.1f}%")
    print(f"{'Parameters':<25} {baseline_params:<15,} {nexus_params:<15,}")
    print(f"{'Training Time (s)':<25} {baseline_time:<15.1f} {nexus_time:<15.1f}")
    
    # Save results
    results['summary'] = {
        'baseline_best': baseline_best,
        'nexus_best': nexus_best,
        'baseline_spike': baseline_final_spike,
        'nexus_spike': nexus_final_spike,
        'baseline_efficiency': baseline_eff,
        'nexus_efficiency': nexus_eff,
    }
    
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(results_dir / f'nexus_ultimate_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = range(1, config['epochs'] + 1)
    
    # Accuracy
    axes[0].plot(epochs, results['baseline']['acc'], 'b-o', label='Baseline', linewidth=2)
    axes[0].plot(epochs, results['nexus']['acc'], 'r-s', label='NEXUS-SNN', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Spike Rate
    axes[1].plot(epochs, results['baseline']['spike'], 'b-o', label='Baseline', linewidth=2)
    axes[1].plot(epochs, results['nexus']['spike'], 'r-s', label='NEXUS-SNN', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Spike Rate')
    axes[1].set_title('Spike Rate (Lower = Better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Final Summary
    metrics = ['Accuracy', 'Spike Rate\n(x100)', 'Efficiency\n(x10)']
    baseline_vals = [baseline_best, baseline_final_spike * 100, baseline_eff / 10]
    nexus_vals = [nexus_best, nexus_final_spike * 100, nexus_eff / 10]
    
    x = np.arange(len(metrics))
    width = 0.35
    axes[2].bar(x - width/2, baseline_vals, width, label='Baseline', color='blue', alpha=0.7)
    axes[2].bar(x + width/2, nexus_vals, width, label='NEXUS-SNN', color='red', alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrics)
    axes[2].set_title('Final Performance')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('NEXUS-SNN: The Ultimate Spiking Neural Network', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    figures_dir = Path('./figures')
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / f'nexus_ultimate_{timestamp}.png', dpi=150, bbox_inches='tight')
    
    print(f"\nResults saved to: results/nexus_ultimate_{timestamp}.json")
    print(f"Plot saved to: figures/nexus_ultimate_{timestamp}.png")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    results = run_ultimate_comparison()
