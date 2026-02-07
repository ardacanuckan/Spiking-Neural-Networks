#!/usr/bin/env python3
"""
DASNN vs Baseline Comparison Experiment

This script trains and compares:
1. DASNN (Dendritic Attention SNN) - Our novel model
2. Baseline LIF SNN - Standard spiking neural network

Using real MNIST dataset to demonstrate the effectiveness of dendritic computation.
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("DASNN vs Baseline SNN Comparison Experiment")
print("Using MNIST Dataset")
print("=" * 70)

# ============================================================================
# Surrogate Gradient Function
# ============================================================================

class ATanSurrogate(torch.autograd.Function):
    """Arctangent surrogate gradient for spike function."""
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = 2.0
        grad = 1.0 / (np.pi * (1 + (np.pi * alpha * x) ** 2))
        return grad * grad_output


def spike_fn(x):
    """Spike function with surrogate gradient."""
    return ATanSurrogate.apply(x)


# ============================================================================
# Baseline LIF Model (Standard SNN)
# ============================================================================

class BaselineLIF(nn.Module):
    """
    Baseline Leaky Integrate-and-Fire SNN.
    
    Standard feedforward SNN without dendritic computation.
    """
    
    def __init__(
        self,
        input_size=784,
        hidden_sizes=[512, 256],
        num_classes=10,
        tau=2.0,
        v_threshold=1.0
    ):
        super().__init__()
        
        self.hidden_sizes = hidden_sizes
        self.v_threshold = v_threshold
        
        # Decay factor
        beta = 1.0 - 1.0 / tau
        self.register_buffer('beta', torch.tensor(beta))
        
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        
        self.fc_out = nn.Linear(hidden_sizes[1], num_classes)
        
        # Membrane potentials
        self.v1 = None
        self.v2 = None
        
    def reset_state(self):
        self.v1 = None
        self.v2 = None
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Layer 1
        current1 = self.bn1(self.fc1(x))
        if self.v1 is None:
            self.v1 = torch.zeros_like(current1)
        self.v1 = self.beta * self.v1 + (1 - self.beta) * current1
        spike1 = spike_fn(self.v1 - self.v_threshold)
        self.v1 = self.v1 - spike1.detach() * self.v_threshold
        
        # Layer 2
        current2 = self.bn2(self.fc2(spike1))
        if self.v2 is None:
            self.v2 = torch.zeros_like(current2)
        self.v2 = self.beta * self.v2 + (1 - self.beta) * current2
        spike2 = spike_fn(self.v2 - self.v_threshold)
        self.v2 = self.v2 - spike2.detach() * self.v_threshold
        
        # Output (no spike, just linear)
        out = self.fc_out(spike2)
        
        return out, (spike1, spike2)


# ============================================================================
# DASNN Model (Our Novel Contribution)
# ============================================================================

class DendriticBranch(nn.Module):
    """
    Dendritic branch with nonlinear gating and sparse activation.
    Key insight: Use gating to SUPPRESS irrelevant inputs, reducing overall spikes.
    """
    
    def __init__(self, in_features, out_features, tau=4.0):
        super().__init__()
        
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # Context-dependent gating - learns to suppress noise
        self.gate_fc = nn.Linear(in_features, out_features, bias=True)
        
        # Initialize gate bias negative for sparse activation
        nn.init.constant_(self.gate_fc.bias, -1.0)
        
        beta = 1.0 - 1.0 / tau
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
        
        self.v_d = None
    
    def reset_state(self):
        self.v_d = None
    
    def forward(self, x):
        if self.v_d is None:
            self.v_d = torch.zeros(x.size(0), self.out_features, device=x.device)
        
        # Feedforward path
        ff = self.linear(x)
        
        # Learnable gate - sigmoid with negative bias promotes sparsity
        gate = torch.sigmoid(self.gate_fc(x))
        
        # Gated dendritic input (suppresses weak/noisy signals)
        gated = ff * gate
        
        # Dendritic membrane with temporal integration
        self.v_d = self.beta * self.v_d + (1 - self.beta) * gated
        
        return self.v_d, gate


class DASNNModel(nn.Module):
    """
    DASNN: Dendritic Attention Spiking Neural Network
    
    Novel architecture with:
    1. Multi-compartment dendritic neurons with gating (reduces spikes)
    2. Heterogeneous time constants (multi-scale temporal processing)
    3. Dendritic attention via competitive inhibition (enhances selectivity)
    4. Sparse activation through learned suppression
    """
    
    def __init__(
        self,
        input_size=784,
        hidden_sizes=[512, 256],
        num_classes=10,
        num_branches=4,
        tau_range=(2.0, 16.0),
        v_threshold=1.0
    ):
        super().__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_branches = num_branches
        self.v_threshold = v_threshold
        
        # Input encoding layer - same as baseline for fair comparison
        self.fc_in = nn.Linear(input_size, hidden_sizes[0])
        self.bn_in = nn.BatchNorm1d(hidden_sizes[0])
        
        # Input layer decay (faster for encoding)
        in_beta = 1.0 - 1.0 / 2.0
        self.register_buffer('in_beta', torch.tensor(in_beta, dtype=torch.float32))
        
        # Dendritic branches with heterogeneous time constants
        tau_min, tau_max = tau_range
        taus = np.linspace(tau_min, tau_max, num_branches)
        branch_dim = hidden_sizes[1] // num_branches
        
        self.branches = nn.ModuleList([
            DendriticBranch(hidden_sizes[0], branch_dim, tau=tau)
            for tau in taus
        ])
        
        # Learnable branch importance (dendritic attention)
        self.branch_attn = nn.Parameter(torch.ones(num_branches, dtype=torch.float32) / num_branches)
        
        # Somatic integration layer with BatchNorm for stability
        self.soma_bn = nn.BatchNorm1d(hidden_sizes[1])
        soma_beta = 1.0 - 1.0 / 2.0
        self.register_buffer('soma_beta', torch.tensor(soma_beta, dtype=torch.float32))
        
        # Higher threshold for output layer = fewer spikes
        self.soma_threshold = v_threshold * 1.2
        
        # Output projection
        self.fc_out = nn.Linear(hidden_sizes[1], num_classes)
        
        # State variables
        self.v_soma = None
        self.v_in = None
    
    def reset_state(self):
        self.v_soma = None
        self.v_in = None
        for branch in self.branches:
            branch.reset_state()
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # === Input encoding (spiking) ===
        current_in = self.bn_in(self.fc_in(x))
        if self.v_in is None:
            self.v_in = torch.zeros_like(current_in)
        self.v_in = self.in_beta * self.v_in + (1 - self.in_beta) * current_in
        spike_in = spike_fn(self.v_in - self.v_threshold)
        self.v_in = self.v_in - spike_in.detach() * self.v_threshold
        
        # === Dendritic processing with gating ===
        branch_outputs = []
        total_gate_activity = 0.0
        
        for branch in self.branches:
            v_d, gate = branch(spike_in)
            branch_outputs.append(v_d)
            total_gate_activity = total_gate_activity + gate.mean()
        
        # Normalize attention weights
        attn_weights = F.softmax(self.branch_attn, dim=0)
        
        # Weighted concatenation of branch outputs
        weighted_branches = []
        for i, v_d in enumerate(branch_outputs):
            weighted_branches.append(v_d * attn_weights[i])
        
        dendrite_out = torch.cat(weighted_branches, dim=-1)
        
        # === Somatic integration ===
        soma_in = self.soma_bn(dendrite_out)
        if self.v_soma is None:
            self.v_soma = torch.zeros_like(soma_in)
        self.v_soma = self.soma_beta * self.v_soma + (1 - self.soma_beta) * soma_in
        
        # Higher threshold = sparser output spikes
        spike_out = spike_fn(self.v_soma - self.soma_threshold)
        self.v_soma = self.v_soma - spike_out.detach() * self.soma_threshold
        
        # === Output ===
        out = self.fc_out(spike_out)
        
        # Return spikes and gate activity for monitoring
        return out, (spike_in, spike_out, attn_weights)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device, time_steps=4):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    total_spikes = 0
    total_neurons = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Reset model state
        model.reset_state()
        
        # Process through time steps
        outputs = []
        all_spikes = []
        
        for t in range(time_steps):
            out, spikes = model(data)
            outputs.append(out)
            all_spikes.append(spikes)
        
        # Average output over time
        output = torch.stack(outputs).mean(dim=0)
        
        # Compute loss
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Count spikes
        for spikes in all_spikes:
            for s in spikes[:2]:  # First two are actual spike tensors
                if isinstance(s, torch.Tensor):
                    total_spikes += s.sum().item()
                    total_neurons += s.numel()
    
    spike_rate = total_spikes / (total_neurons + 1e-8)
    
    return total_loss / len(train_loader), 100. * correct / total, spike_rate


@torch.no_grad()
def evaluate(model, test_loader, criterion, device, time_steps=4):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_spikes = 0
    total_neurons = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        model.reset_state()
        
        outputs = []
        all_spikes = []
        
        for t in range(time_steps):
            out, spikes = model(data)
            outputs.append(out)
            all_spikes.append(spikes)
        
        output = torch.stack(outputs).mean(dim=0)
        
        total_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        for spikes in all_spikes:
            for s in spikes[:2]:
                if isinstance(s, torch.Tensor):
                    total_spikes += s.sum().item()
                    total_neurons += s.numel()
    
    spike_rate = total_spikes / (total_neurons + 1e-8)
    
    return total_loss / len(test_loader), 100. * correct / total, spike_rate


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment():
    """Run the full comparison experiment."""
    
    # Configuration
    config = {
        'batch_size': 128,
        'epochs': 15,
        'time_steps': 4,
        'learning_rate': 1e-3,
        'hidden_sizes': [512, 256],
        'num_branches': 4,
        'tau_range': (2.0, 16.0),
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Data loading
    print("\n" + "-" * 50)
    print("Loading MNIST Dataset...")
    print("-" * 50)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Results storage
    results = {
        'config': config,
        'baseline': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'spike_rate': []},
        'dasnn': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'spike_rate': []}
    }
    
    # =====================================================================
    # Train Baseline Model
    # =====================================================================
    print("\n" + "=" * 70)
    print("Training BASELINE LIF Model (Standard SNN)")
    print("=" * 70)
    
    baseline_model = BaselineLIF(
        input_size=784,
        hidden_sizes=config['hidden_sizes'],
        num_classes=10,
        tau=2.0,
        v_threshold=1.0
    ).to(device)
    
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"Baseline parameters: {baseline_params:,}")
    
    baseline_optimizer = optim.AdamW(baseline_model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    baseline_scheduler = optim.lr_scheduler.CosineAnnealingLR(baseline_optimizer, config['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    baseline_start_time = time.time()
    
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_spike = train_epoch(
            baseline_model, train_loader, baseline_optimizer, criterion, 
            device, config['time_steps']
        )
        test_loss, test_acc, test_spike = evaluate(
            baseline_model, test_loader, criterion, device, config['time_steps']
        )
        baseline_scheduler.step()
        
        results['baseline']['train_loss'].append(train_loss)
        results['baseline']['train_acc'].append(train_acc)
        results['baseline']['test_loss'].append(test_loss)
        results['baseline']['test_acc'].append(test_acc)
        results['baseline']['spike_rate'].append(test_spike)
        
        print(f"Epoch {epoch+1:2d}/{config['epochs']} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | "
              f"Spike Rate: {test_spike:.4f}")
    
    baseline_time = time.time() - baseline_start_time
    baseline_best_acc = max(results['baseline']['test_acc'])
    
    print(f"\nBaseline Best Accuracy: {baseline_best_acc:.2f}%")
    print(f"Baseline Training Time: {baseline_time:.1f}s")
    
    # =====================================================================
    # Train DASNN Model
    # =====================================================================
    print("\n" + "=" * 70)
    print("Training DASNN Model (Dendritic Attention SNN)")
    print("=" * 70)
    
    dasnn_model = DASNNModel(
        input_size=784,
        hidden_sizes=config['hidden_sizes'],
        num_classes=10,
        num_branches=config['num_branches'],
        tau_range=config['tau_range'],
        v_threshold=1.0
    ).to(device)
    
    dasnn_params = sum(p.numel() for p in dasnn_model.parameters())
    print(f"DASNN parameters: {dasnn_params:,}")
    print(f"Parameter increase: {(dasnn_params - baseline_params) / baseline_params * 100:.1f}%")
    
    dasnn_optimizer = optim.AdamW(dasnn_model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    dasnn_scheduler = optim.lr_scheduler.CosineAnnealingLR(dasnn_optimizer, config['epochs'])
    
    dasnn_start_time = time.time()
    
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_spike = train_epoch(
            dasnn_model, train_loader, dasnn_optimizer, criterion,
            device, config['time_steps']
        )
        test_loss, test_acc, test_spike = evaluate(
            dasnn_model, test_loader, criterion, device, config['time_steps']
        )
        dasnn_scheduler.step()
        
        results['dasnn']['train_loss'].append(train_loss)
        results['dasnn']['train_acc'].append(train_acc)
        results['dasnn']['test_loss'].append(test_loss)
        results['dasnn']['test_acc'].append(test_acc)
        results['dasnn']['spike_rate'].append(test_spike)
        
        print(f"Epoch {epoch+1:2d}/{config['epochs']} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | "
              f"Spike Rate: {test_spike:.4f}")
    
    dasnn_time = time.time() - dasnn_start_time
    dasnn_best_acc = max(results['dasnn']['test_acc'])
    
    print(f"\nDASNN Best Accuracy: {dasnn_best_acc:.2f}%")
    print(f"DASNN Training Time: {dasnn_time:.1f}s")
    
    # =====================================================================
    # Comparison Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'DASNN':<15} {'Improvement':<15}")
    print("-" * 70)
    
    acc_improvement = dasnn_best_acc - baseline_best_acc
    print(f"{'Best Test Accuracy':<25} {baseline_best_acc:<15.2f} {dasnn_best_acc:<15.2f} {acc_improvement:+.2f}%")
    
    baseline_final_spike = results['baseline']['spike_rate'][-1]
    dasnn_final_spike = results['dasnn']['spike_rate'][-1]
    spike_reduction = (baseline_final_spike - dasnn_final_spike) / baseline_final_spike * 100
    print(f"{'Final Spike Rate':<25} {baseline_final_spike:<15.4f} {dasnn_final_spike:<15.4f} {spike_reduction:+.1f}% reduction")
    
    print(f"{'Parameters':<25} {baseline_params:<15,} {dasnn_params:<15,} {dasnn_params - baseline_params:+,}")
    print(f"{'Training Time (s)':<25} {baseline_time:<15.1f} {dasnn_time:<15.1f} {dasnn_time - baseline_time:+.1f}s")
    
    # Energy efficiency metric (spikes * accuracy trade-off)
    baseline_efficiency = baseline_best_acc / (baseline_final_spike + 0.01)
    dasnn_efficiency = dasnn_best_acc / (dasnn_final_spike + 0.01)
    efficiency_improvement = (dasnn_efficiency - baseline_efficiency) / baseline_efficiency * 100
    print(f"{'Energy Efficiency Score':<25} {baseline_efficiency:<15.1f} {dasnn_efficiency:<15.1f} {efficiency_improvement:+.1f}%")
    
    # =====================================================================
    # Save Results
    # =====================================================================
    results['summary'] = {
        'baseline_best_acc': baseline_best_acc,
        'dasnn_best_acc': dasnn_best_acc,
        'accuracy_improvement': acc_improvement,
        'baseline_spike_rate': baseline_final_spike,
        'dasnn_spike_rate': dasnn_final_spike,
        'spike_reduction_percent': spike_reduction,
        'baseline_params': baseline_params,
        'dasnn_params': dasnn_params,
        'baseline_time': baseline_time,
        'dasnn_time': dasnn_time,
        'baseline_efficiency': baseline_efficiency,
        'dasnn_efficiency': dasnn_efficiency
    }
    
    # Save to file
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'comparison_results_{timestamp}.json'
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # =====================================================================
    # Plot Results
    # =====================================================================
    print("\nGenerating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, config['epochs'] + 1)
    
    # Test Accuracy
    ax1 = axes[0, 0]
    ax1.plot(epochs, results['baseline']['test_acc'], 'b-o', label='Baseline LIF', linewidth=2, markersize=4)
    ax1.plot(epochs, results['dasnn']['test_acc'], 'r-s', label='DASNN (Ours)', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training Loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, results['baseline']['train_loss'], 'b-o', label='Baseline LIF', linewidth=2, markersize=4)
    ax2.plot(epochs, results['dasnn']['train_loss'], 'r-s', label='DASNN (Ours)', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Spike Rate
    ax3 = axes[1, 0]
    ax3.plot(epochs, results['baseline']['spike_rate'], 'b-o', label='Baseline LIF', linewidth=2, markersize=4)
    ax3.plot(epochs, results['dasnn']['spike_rate'], 'r-s', label='DASNN (Ours)', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Spike Rate')
    ax3.set_title('Spike Rate Comparison (Lower = More Efficient)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final Summary Bar Chart
    ax4 = axes[1, 1]
    metrics = ['Accuracy\n(%)', 'Spike Rate\n(×100)', 'Efficiency\n(×10)']
    baseline_vals = [baseline_best_acc, baseline_final_spike * 100, baseline_efficiency / 10]
    dasnn_vals = [dasnn_best_acc, dasnn_final_spike * 100, dasnn_efficiency / 10]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, baseline_vals, width, label='Baseline LIF', color='blue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, dasnn_vals, width, label='DASNN (Ours)', color='red', alpha=0.7)
    
    ax4.set_ylabel('Value')
    ax4.set_title('Final Performance Summary')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, baseline_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, dasnn_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('DASNN vs Baseline SNN Comparison on MNIST', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    figures_dir = Path('./figures')
    figures_dir.mkdir(exist_ok=True)
    plot_file = figures_dir / f'comparison_plot_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = run_experiment()
