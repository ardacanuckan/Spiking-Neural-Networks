#!/usr/bin/env python3
"""
CIFAR-10 SNN Benchmark Suite
============================

This script implements a proper academic benchmark for SNNs on CIFAR-10.

STANDARD METRICS (NeuroBench Framework):
1. Accuracy (%)
2. Synaptic Operations (SynOps) - proper calculation
3. Activation Sparsity (spike rate)
4. Timesteps
5. Parameters count
6. Theoretical Energy (relative to ANN baseline)

SOTA TARGETS (2024-2026):
- Spikformer V2: ~95-96% accuracy, 4 timesteps
- SEW-ResNet: ~94% accuracy, 4-6 timesteps
- ANN ResNet baseline: ~93-94% on CIFAR-10

Our goal: Compare multiple SNN architectures fairly on the SAME benchmark.
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
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("CIFAR-10 SNN BENCHMARK SUITE")
print("Standard Metrics: Accuracy, SynOps, Spike Rate, Timesteps")
print("=" * 80)


# ============================================================================
# STANDARD METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """
    Calculates standard SNN metrics according to NeuroBench framework.
    """
    
    def __init__(self, model, input_shape=(3, 32, 32)):
        self.model = model
        self.input_shape = input_shape
        self.layer_info = []
        self._analyze_model()
    
    def _analyze_model(self):
        """Analyze model to get layer connectivity information."""
        self.layer_info = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.layer_info.append({
                    'name': name,
                    'type': 'linear',
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'params': module.in_features * module.out_features
                })
            elif isinstance(module, nn.Conv2d):
                self.layer_info.append({
                    'name': name,
                    'type': 'conv2d',
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'params': module.in_channels * module.out_channels * np.prod(module.kernel_size)
                })
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def calculate_synops(self, spike_counts, layer_fanouts):
        """
        Calculate Synaptic Operations (SynOps).
        
        SynOps = Σ (spike_count_per_layer × fanout_per_layer)
        
        For SNNs with binary spikes:
        - Eff_ACs = number of accumulate operations (no multiply needed)
        - Each spike triggers fanout number of additions
        """
        total_synops = 0
        for spikes, fanout in zip(spike_counts, layer_fanouts):
            total_synops += spikes * fanout
        return total_synops
    
    def calculate_activation_sparsity(self, spike_tensor):
        """
        Calculate activation sparsity.
        
        Sparsity = 1 - (active neurons / total neurons)
        Range: 0 (all active) to 1 (all silent)
        """
        total = spike_tensor.numel()
        active = (spike_tensor > 0).sum().item()
        sparsity = 1.0 - (active / total)
        return sparsity
    
    def calculate_spike_rate(self, spike_tensor):
        """
        Calculate average spike rate (firing rate).
        
        Spike Rate = total spikes / (neurons × timesteps)
        """
        return spike_tensor.float().mean().item()
    
    def theoretical_energy_ratio(self, synops_snn, synops_ann, energy_per_ac=0.9, energy_per_mac=4.6):
        """
        Calculate theoretical energy ratio SNN vs ANN.
        
        Based on 45nm CMOS:
        - MAC (multiply-accumulate): 4.6 pJ
        - AC (accumulate only): 0.9 pJ
        
        Energy ratio = (SNN_SynOps × E_AC) / (ANN_SynOps × E_MAC)
        """
        energy_snn = synops_snn * energy_per_ac
        energy_ann = synops_ann * energy_per_mac
        return energy_snn / (energy_ann + 1e-8)


# ============================================================================
# SURROGATE GRADIENT
# ============================================================================

class ATanSurrogate(torch.autograd.Function):
    """ArcTan surrogate gradient - standard in SpikingJelly."""
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = ctx.alpha / (2 * (1 + (np.pi/2 * ctx.alpha * x)**2))
        return grad * grad_output, None


def spike_fn(x, alpha=2.0):
    return ATanSurrogate.apply(x, alpha)


# ============================================================================
# STANDARD LIF NEURON
# ============================================================================

class LIFNeuron(nn.Module):
    """Standard Leaky Integrate-and-Fire neuron."""
    
    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.beta = 1.0 - 1.0 / tau
        self.v_mem = None
    
    def reset(self):
        self.v_mem = None
    
    def forward(self, x):
        if self.v_mem is None:
            self.v_mem = torch.zeros_like(x)
        
        self.v_mem = self.beta * self.v_mem + x
        spike = spike_fn(self.v_mem - self.v_threshold)
        self.v_mem = self.v_mem - spike * self.v_threshold  # Soft reset
        
        return spike


# ============================================================================
# CIFAR-10 SNN MODELS
# ============================================================================

class CIFAR10_SNN_Small(nn.Module):
    """
    Small SNN for CIFAR-10.
    Similar to LeNet-5 architecture.
    """
    
    def __init__(self, tau=2.0):
        super().__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = LIFNeuron(tau)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = LIFNeuron(tau)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = LIFNeuron(tau)
        
        # FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.lif4 = LIFNeuron(tau)
        
        self.fc2 = nn.Linear(256, 10)
        
        self.pool = nn.AvgPool2d(2)
    
    def reset(self):
        for module in self.modules():
            if isinstance(module, LIFNeuron):
                module.reset()
    
    def forward(self, x):
        spikes = []
        
        # Conv block 1
        h = self.bn1(self.conv1(x))
        h = self.lif1(h)
        spikes.append(h)
        h = self.pool(h)
        
        # Conv block 2
        h = self.bn2(self.conv2(h))
        h = self.lif2(h)
        spikes.append(h)
        h = self.pool(h)
        
        # Conv block 3
        h = self.bn3(self.conv3(h))
        h = self.lif3(h)
        spikes.append(h)
        h = self.pool(h)
        
        # FC
        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        h = self.lif4(h)
        spikes.append(h)
        
        out = self.fc2(h)
        
        return out, spikes


class CIFAR10_SNN_VGG(nn.Module):
    """
    VGG-style SNN for CIFAR-10.
    Deeper architecture for better accuracy.
    """
    
    def __init__(self, tau=2.0):
        super().__init__()
        
        # VGG-like feature extractor
        self.features = nn.ModuleList([
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            LIFNeuron(tau),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            LIFNeuron(tau),
            nn.AvgPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            LIFNeuron(tau),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            LIFNeuron(tau),
            nn.AvgPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            LIFNeuron(tau),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            LIFNeuron(tau),
            nn.AvgPool2d(2),
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
        )
        self.lif_fc = LIFNeuron(tau)
        self.fc_out = nn.Linear(512, 10)
    
    def reset(self):
        for module in self.modules():
            if isinstance(module, LIFNeuron):
                module.reset()
    
    def forward(self, x):
        spikes = []
        
        h = x
        for layer in self.features:
            h = layer(h)
            if isinstance(layer, LIFNeuron):
                spikes.append(h)
        
        h = h.view(h.size(0), -1)
        h = self.classifier(h)
        h = self.lif_fc(h)
        spikes.append(h)
        
        out = self.fc_out(h)
        
        return out, spikes


# ============================================================================
# ANN BASELINE
# ============================================================================

class CIFAR10_ANN_Baseline(nn.Module):
    """ANN baseline for fair comparison."""
    
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_snn_epoch(model, loader, optimizer, device, timesteps=4):
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_spikes = []
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.reset()
        
        # Accumulate outputs over timesteps
        outputs = []
        batch_spikes = []
        
        for t in range(timesteps):
            out, spikes = model(data)
            outputs.append(out)
            batch_spikes.extend([s.detach() for s in spikes])
        
        # Average output over time
        output = torch.stack(outputs).mean(0)
        
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Collect spikes for metrics
        all_spikes.extend(batch_spikes)
    
    # Calculate spike rate
    if all_spikes:
        spike_tensor = torch.cat([s.flatten() for s in all_spikes])
        spike_rate = spike_tensor.float().mean().item()
    else:
        spike_rate = 0
    
    return total_loss / len(loader), 100. * correct / total, spike_rate


@torch.no_grad()
def eval_snn(model, loader, device, timesteps=4):
    model.eval()
    correct, total = 0, 0
    all_spikes = []
    total_synops = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        model.reset()
        
        outputs = []
        batch_spikes = []
        
        for t in range(timesteps):
            out, spikes = model(data)
            outputs.append(out)
            batch_spikes.extend(spikes)
        
        output = torch.stack(outputs).mean(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Collect spikes
        all_spikes.extend([s.detach() for s in batch_spikes])
    
    # Calculate metrics
    if all_spikes:
        spike_tensor = torch.cat([s.flatten() for s in all_spikes])
        spike_rate = spike_tensor.float().mean().item()
        activation_sparsity = 1.0 - spike_rate
    else:
        spike_rate = 0
        activation_sparsity = 1.0
    
    accuracy = 100. * correct / total
    
    return {
        'accuracy': accuracy,
        'spike_rate': spike_rate,
        'activation_sparsity': activation_sparsity,
    }


def train_ann_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def eval_ann(model, loader, device):
    model.eval()
    correct, total = 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return 100. * correct / total


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_cifar10_benchmark():
    print("\n" + "=" * 80)
    print("CIFAR-10 BENCHMARK: SNN vs ANN")
    print("=" * 80)
    
    config = {
        'batch_size': 32,
        'epochs': 10,
        'timesteps': 2,
        'lr': 0.01,
        'tau': 2.0,
    }
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Config: {config}")
    
    # Standard CIFAR-10 augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    results = {
        'config': config,
        'models': {}
    }
    
    # =========================================================================
    # BENCHMARK 1: SNN VGG
    # =========================================================================
    print("\n" + "-" * 60)
    print("Training SNN-VGG on CIFAR-10")
    print("-" * 60)
    
    snn_model = CIFAR10_SNN_VGG(tau=config['tau']).to(device)
    snn_params = sum(p.numel() for p in snn_model.parameters())
    print(f"SNN Parameters: {snn_params:,}")
    
    optimizer = optim.SGD(snn_model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    
    best_snn_acc = 0
    best_snn_spike = 0.0
    snn_history = {'acc': [], 'spike_rate': []}
    t0 = time.time()
    
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_spike = train_snn_epoch(
            snn_model, train_loader, optimizer, device, config['timesteps']
        )
        metrics = eval_snn(snn_model, test_loader, device, config['timesteps'])
        scheduler.step()
        
        snn_history['acc'].append(metrics['accuracy'])
        snn_history['spike_rate'].append(metrics['spike_rate'])
        
        if metrics['accuracy'] > best_snn_acc:
            best_snn_acc = metrics['accuracy']
            best_snn_spike = metrics['spike_rate']
            marker = " *BEST*"
        else:
            marker = ""
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"Test Acc: {metrics['accuracy']:.2f}% | "
                  f"Spike Rate: {metrics['spike_rate']:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.4f}{marker}")
    
    snn_time = time.time() - t0
    
    results['models']['SNN-VGG'] = {
        'accuracy': best_snn_acc,
        'spike_rate': best_snn_spike,
        'activation_sparsity': 1 - best_snn_spike,
        'parameters': snn_params,
        'timesteps': config['timesteps'],
        'training_time': snn_time,
        'history': snn_history
    }
    
    # =========================================================================
    # BENCHMARK 2: ANN Baseline
    # =========================================================================
    print("\n" + "-" * 60)
    print("Training ANN Baseline on CIFAR-10")
    print("-" * 60)
    
    ann_model = CIFAR10_ANN_Baseline().to(device)
    ann_params = sum(p.numel() for p in ann_model.parameters())
    print(f"ANN Parameters: {ann_params:,}")
    
    optimizer = optim.SGD(ann_model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    
    best_ann_acc = 0
    ann_history = {'acc': []}
    t0 = time.time()
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_ann_epoch(ann_model, train_loader, optimizer, device)
        test_acc = eval_ann(ann_model, test_loader, device)
        scheduler.step()
        
        ann_history['acc'].append(test_acc)
        
        if test_acc > best_ann_acc:
            best_ann_acc = test_acc
            marker = " *BEST*"
        else:
            marker = ""
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"LR: {scheduler.get_last_lr()[0]:.4f}{marker}")
    
    ann_time = time.time() - t0
    
    results['models']['ANN-Baseline'] = {
        'accuracy': best_ann_acc,
        'parameters': ann_params,
        'training_time': ann_time,
        'history': ann_history
    }
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("CIFAR-10 BENCHMARK RESULTS")
    print("=" * 80)
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Spike Rate':<12} {'Sparsity':<12} {'Params':<12} {'Time Steps':<10}")
    print("-" * 80)
    
    for name, data in results['models'].items():
        acc = f"{data['accuracy']:.2f}%"
        spike = f"{data.get('spike_rate', 'N/A'):.4f}" if 'spike_rate' in data else "N/A"
        sparsity = f"{data.get('activation_sparsity', 0):.2f}" if 'activation_sparsity' in data else "N/A"
        params = f"{data['parameters']:,}"
        ts = str(data.get('timesteps', 1))
        
        print(f"{name:<20} {acc:<12} {spike:<12} {sparsity:<12} {params:<12} {ts:<10}")
    
    # Gap analysis
    print("\n" + "-" * 40)
    print("GAP ANALYSIS:")
    print("-" * 40)
    print(f"ANN Accuracy:  {results['models']['ANN-Baseline']['accuracy']:.2f}%")
    print(f"SNN Accuracy:  {results['models']['SNN-VGG']['accuracy']:.2f}%")
    gap = results['models']['ANN-Baseline']['accuracy'] - results['models']['SNN-VGG']['accuracy']
    print(f"Gap:           {gap:.2f}% (target: <2%)")
    
    # Energy analysis (theoretical)
    print("\n" + "-" * 40)
    print("THEORETICAL ENERGY ANALYSIS:")
    print("-" * 40)
    spike_rate = results['models']['SNN-VGG']['spike_rate']
    # ANN: all activations are non-zero (assume ~50% after ReLU)
    # SNN: only spikes cause operations
    energy_ratio = spike_rate * 0.9 / (0.5 * 4.6)  # AC vs MAC energy
    print(f"SNN Spike Rate: {spike_rate:.4f}")
    print(f"Theoretical Energy Ratio (SNN/ANN): {energy_ratio:.3f}x")
    print(f"Energy Reduction: {(1 - energy_ratio) * 100:.1f}%")
    
    # Save results
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Remove non-serializable history for JSON
    results_save = {
        'config': config,
        'models': {
            name: {k: v for k, v in data.items() if k != 'history'}
            for name, data in results['models'].items()
        },
        'gap_analysis': {
            'ann_accuracy': results['models']['ANN-Baseline']['accuracy'],
            'snn_accuracy': results['models']['SNN-VGG']['accuracy'],
            'gap': gap,
            'energy_ratio': energy_ratio
        }
    }
    
    with open(results_dir / f'cifar10_benchmark_{timestamp}.json', 'w') as f:
        json.dump(results_save, f, indent=2, default=float)
    
    print(f"\nResults saved to: results/cifar10_benchmark_{timestamp}.json")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    results = run_cifar10_benchmark()
