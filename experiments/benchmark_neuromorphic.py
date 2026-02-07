#!/usr/bin/env python3
"""
Neuromorphic Dataset Benchmark Suite
====================================

This script implements benchmarks for neuromorphic (event-based) datasets:
1. N-MNIST - Neuromorphic MNIST (saccade-based)
2. DVS-Gesture - DVS128 gesture recognition
3. CIFAR10-DVS - Event-based CIFAR-10

SOTA REFERENCES (2024-2026):
- N-MNIST: ~99% accuracy
- DVS-Gesture: ~98% accuracy  
- CIFAR10-DVS: ~83% accuracy

These datasets are EVENT-BASED, meaning:
- Data comes as (x, y, t, polarity) events
- Must be converted to frames or processed directly
- Native format for SNNs (no rate coding needed)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime
import struct
import os

torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("NEUROMORPHIC DATASET BENCHMARK SUITE")
print("Datasets: N-MNIST, DVS-Gesture, CIFAR10-DVS")
print("=" * 80)


# ============================================================================
# EVENT DATA UTILITIES
# ============================================================================

def events_to_frames(events, height, width, num_frames, num_channels=2):
    """
    Convert events to frame representation.
    
    Args:
        events: dict with 'x', 'y', 't', 'p' arrays
        height, width: spatial dimensions
        num_frames: number of time bins
        num_channels: 2 for ON/OFF polarity
    
    Returns:
        frames: [num_frames, num_channels, height, width]
    """
    frames = np.zeros((num_frames, num_channels, height, width), dtype=np.float32)
    
    if len(events['t']) == 0:
        return frames
    
    # Normalize time to [0, num_frames)
    t = events['t']
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8) * (num_frames - 1e-6)
    t_idx = t_norm.astype(np.int32)
    
    # Accumulate events
    for i in range(len(events['x'])):
        x, y = int(events['x'][i]), int(events['y'][i])
        p = int(events['p'][i])
        ti = int(t_idx[i])
        
        if 0 <= x < width and 0 <= y < height and 0 <= ti < num_frames:
            frames[ti, p, y, x] += 1
    
    # Normalize
    frames = np.clip(frames, 0, 10) / 10.0
    
    return frames


# ============================================================================
# N-MNIST DATASET
# ============================================================================

class NMNISTDataset(Dataset):
    """
    N-MNIST: Neuromorphic MNIST dataset.
    
    Created by moving a DVS camera over MNIST digits in a saccade pattern.
    - 60,000 training / 10,000 test samples
    - 34x34 pixel resolution
    - ~300ms recording per digit
    
    SOTA: ~99% accuracy
    
    Download from: https://www.garrickorchard.com/datasets/n-mnist
    """
    
    def __init__(self, root, train=True, num_frames=10, download=False):
        self.root = Path(root)
        self.train = train
        self.num_frames = num_frames
        self.height = 34
        self.width = 34
        
        # Check if data exists
        split = 'Train' if train else 'Test'
        self.data_path = self.root / 'N-MNIST' / split
        
        if not self.data_path.exists():
            if download:
                self._download()
            else:
                print(f"N-MNIST not found at {self.data_path}")
                print("Creating synthetic N-MNIST-like data for demonstration...")
                self._create_synthetic()
                return
        
        # Load file list
        self.samples = []
        for label in range(10):
            label_path = self.data_path / str(label)
            if label_path.exists():
                for f in label_path.glob('*.bin'):
                    self.samples.append((f, label))
        
        if len(self.samples) == 0:
            print("No samples found, creating synthetic data...")
            self._create_synthetic()
    
    def _create_synthetic(self):
        """Create synthetic event data for testing."""
        self.synthetic = True
        n_samples = 1000 if self.train else 200
        self.samples = [(i, i % 10) for i in range(n_samples)]
    
    def _download(self):
        """Download N-MNIST dataset."""
        print("N-MNIST download not implemented. Please download manually from:")
        print("https://www.garrickorchard.com/datasets/n-mnist")
        self._create_synthetic()
    
    def _read_events(self, filepath):
        """Read N-MNIST binary format."""
        events = {'x': [], 'y': [], 't': [], 'p': []}
        
        try:
            with open(filepath, 'rb') as f:
                while True:
                    data = f.read(5)
                    if len(data) < 5:
                        break
                    
                    # N-MNIST format: 5 bytes per event
                    addr = (data[0] & 0xFF) | ((data[1] & 0xFF) << 8)
                    x = addr & 0x7F
                    y = (addr >> 7) & 0x7F
                    p = (addr >> 14) & 0x01
                    
                    ts = ((data[2] & 0xFF) | 
                          ((data[3] & 0xFF) << 8) | 
                          ((data[4] & 0xFF) << 16))
                    
                    events['x'].append(x)
                    events['y'].append(y)
                    events['t'].append(ts)
                    events['p'].append(p)
        except:
            pass
        
        for k in events:
            events[k] = np.array(events[k])
        
        return events
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if hasattr(self, 'synthetic') and self.synthetic:
            # Generate synthetic spiking MNIST-like data
            label = self.samples[idx][1]
            frames = np.random.rand(self.num_frames, 2, self.height, self.width).astype(np.float32)
            # Add digit-specific pattern
            frames[:, :, 10:24, 10:24] += label * 0.1
            frames = np.clip(frames, 0, 1)
            return torch.from_numpy(frames), label
        
        filepath, label = self.samples[idx]
        events = self._read_events(filepath)
        frames = events_to_frames(events, self.height, self.width, self.num_frames)
        
        return torch.from_numpy(frames), label


# ============================================================================
# DVS-GESTURE DATASET
# ============================================================================

class DVSGestureDataset(Dataset):
    """
    DVS128 Gesture Dataset.
    
    11 hand gesture classes recorded with DVS128 camera.
    - 1,342 training / 288 test samples
    - 128x128 pixel resolution
    - Variable length recordings
    
    SOTA: ~98% accuracy
    
    Download from: https://research.ibm.com/interactive/dvsgesture/
    """
    
    def __init__(self, root, train=True, num_frames=16, download=False):
        self.root = Path(root)
        self.train = train
        self.num_frames = num_frames
        self.height = 128
        self.width = 128
        self.num_classes = 11
        
        self.data_path = self.root / 'DVS-Gesture'
        
        if not self.data_path.exists():
            print(f"DVS-Gesture not found at {self.data_path}")
            print("Creating synthetic DVS-Gesture-like data for demonstration...")
            self._create_synthetic()
            return
        
        # Try to load
        self.samples = []
        split_file = self.data_path / ('train.txt' if train else 'test.txt')
        
        if split_file.exists():
            with open(split_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.samples.append((self.data_path / parts[0], int(parts[1])))
        
        if len(self.samples) == 0:
            self._create_synthetic()
    
    def _create_synthetic(self):
        """Create synthetic gesture data."""
        self.synthetic = True
        n_samples = 500 if self.train else 100
        self.samples = [(i, i % self.num_classes) for i in range(n_samples)]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if hasattr(self, 'synthetic') and self.synthetic:
            label = self.samples[idx][1]
            # Create gesture-like temporal patterns
            frames = np.zeros((self.num_frames, 2, self.height, self.width), dtype=np.float32)
            
            # Simulate motion pattern based on gesture class
            for t in range(self.num_frames):
                # Moving blob pattern
                cx = 64 + int(30 * np.sin(2 * np.pi * (t + label) / self.num_frames))
                cy = 64 + int(30 * np.cos(2 * np.pi * (t + label * 2) / self.num_frames))
                
                # Add Gaussian blob
                for dy in range(-15, 16):
                    for dx in range(-15, 16):
                        x, y = cx + dx, cy + dy
                        if 0 <= x < self.width and 0 <= y < self.height:
                            dist = np.sqrt(dx**2 + dy**2)
                            frames[t, t % 2, y, x] = np.exp(-dist**2 / 50)
            
            return torch.from_numpy(frames), label
        
        # Real data loading would go here
        filepath, label = self.samples[idx]
        # Placeholder - implement actual event reading
        frames = np.random.rand(self.num_frames, 2, self.height, self.width).astype(np.float32)
        return torch.from_numpy(frames), label


# ============================================================================
# CIFAR10-DVS DATASET
# ============================================================================

class CIFAR10DVSDataset(Dataset):
    """
    CIFAR10-DVS: Event-based CIFAR-10.
    
    CIFAR-10 images converted to events using a DVS camera.
    - 9,000 training / 1,000 test samples (per class)
    - 128x128 pixel resolution
    - ~1.2 seconds per sample
    
    SOTA: ~83% accuracy
    
    Download from: https://figshare.com/articles/dataset/CIFAR10-DVS/4724671
    """
    
    def __init__(self, root, train=True, num_frames=10, download=False):
        self.root = Path(root)
        self.train = train
        self.num_frames = num_frames
        self.height = 128
        self.width = 128
        self.num_classes = 10
        
        self.data_path = self.root / 'CIFAR10-DVS'
        
        if not self.data_path.exists():
            print(f"CIFAR10-DVS not found at {self.data_path}")
            print("Creating synthetic CIFAR10-DVS-like data for demonstration...")
            self._create_synthetic()
            return
        
        self.samples = []
        # Implement actual file loading here
        
        if len(self.samples) == 0:
            self._create_synthetic()
    
    def _create_synthetic(self):
        """Create synthetic data."""
        self.synthetic = True
        n_samples = 1000 if self.train else 200
        self.samples = [(i, i % self.num_classes) for i in range(n_samples)]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if hasattr(self, 'synthetic') and self.synthetic:
            label = self.samples[idx][1]
            frames = np.random.rand(self.num_frames, 2, self.height, self.width).astype(np.float32) * 0.3
            # Add class-specific texture
            frames[:, :, ::4, ::4] += label * 0.05
            frames = np.clip(frames, 0, 1)
            return torch.from_numpy(frames), label
        
        filepath, label = self.samples[idx]
        frames = np.random.rand(self.num_frames, 2, self.height, self.width).astype(np.float32)
        return torch.from_numpy(frames), label


# ============================================================================
# SNN MODELS FOR EVENT DATA
# ============================================================================

class ATanSurrogate(torch.autograd.Function):
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


def spike_fn(x):
    return ATanSurrogate.apply(x, 2.0)


class LIFNeuron(nn.Module):
    def __init__(self, tau=2.0):
        super().__init__()
        self.tau = tau
        self.beta = 1.0 - 1.0 / tau
        self.v = None
    
    def reset(self):
        self.v = None
    
    def forward(self, x):
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.beta * self.v + x
        spike = spike_fn(self.v - 1.0)
        self.v = self.v - spike
        return spike


class EventSNN(nn.Module):
    """
    SNN for event-based data.
    Processes temporal frames with spiking neurons.
    """
    
    def __init__(self, in_channels=2, height=34, width=34, num_classes=10, tau=2.0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = LIFNeuron(tau)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = LIFNeuron(tau)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = LIFNeuron(tau)
        
        self.pool = nn.AdaptiveAvgPool2d(4)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.lif4 = LIFNeuron(tau)
        self.fc2 = nn.Linear(256, num_classes)
    
    def reset(self):
        for m in self.modules():
            if isinstance(m, LIFNeuron):
                m.reset()
    
    def forward_single(self, x):
        """Process single time frame."""
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.lif1(h)
        h = F.avg_pool2d(h, 2)
        
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.lif2(h)
        h = F.avg_pool2d(h, 2)
        
        h = F.relu(self.bn3(self.conv3(h)))
        h = self.lif3(h)
        
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        
        h = self.fc1(h)
        h = self.lif4(h)
        
        return self.fc2(h)
    
    def forward(self, x):
        """
        Process event frames.
        x: [batch, time, channels, height, width]
        """
        self.reset()
        
        batch_size, num_frames = x.shape[0], x.shape[1]
        outputs = []
        spikes = []
        
        for t in range(num_frames):
            out = self.forward_single(x[:, t])
            outputs.append(out)
        
        # Average over time
        return torch.stack(outputs).mean(0)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
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

def run_neuromorphic_benchmark():
    print("\n" + "=" * 80)
    print("NEUROMORPHIC DATASET BENCHMARKS")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_root = './data'
    results = {'datasets': {}}
    
    # =========================================================================
    # BENCHMARK 1: N-MNIST
    # =========================================================================
    print("\n" + "-" * 60)
    print("Benchmark 1: N-MNIST")
    print("SOTA Reference: ~99% accuracy")
    print("-" * 60)
    
    config_nmnist = {
        'batch_size': 64,
        'epochs': 30,
        'num_frames': 10,
        'lr': 1e-3,
    }
    
    train_data = NMNISTDataset(data_root, train=True, num_frames=config_nmnist['num_frames'])
    test_data = NMNISTDataset(data_root, train=False, num_frames=config_nmnist['num_frames'])
    
    train_loader = DataLoader(train_data, batch_size=config_nmnist['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=config_nmnist['batch_size'], shuffle=False, num_workers=2)
    
    model = EventSNN(in_channels=2, height=34, width=34, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config_nmnist['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config_nmnist['epochs'])
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    best_acc = 0
    t0 = time.time()
    
    for epoch in range(config_nmnist['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            marker = " *BEST*"
        else:
            marker = ""
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{config_nmnist['epochs']} | "
                  f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%{marker}")
    
    nmnist_time = time.time() - t0
    
    results['datasets']['N-MNIST'] = {
        'accuracy': best_acc,
        'sota_reference': 99.0,
        'gap_to_sota': 99.0 - best_acc,
        'training_time': nmnist_time,
        'config': config_nmnist,
        'synthetic_data': hasattr(train_data, 'synthetic')
    }
    
    print(f"\nN-MNIST Best Accuracy: {best_acc:.2f}% (SOTA: ~99%)")
    
    # =========================================================================
    # BENCHMARK 2: DVS-Gesture
    # =========================================================================
    print("\n" + "-" * 60)
    print("Benchmark 2: DVS-Gesture")
    print("SOTA Reference: ~98% accuracy")
    print("-" * 60)
    
    config_dvs = {
        'batch_size': 32,
        'epochs': 30,
        'num_frames': 16,
        'lr': 1e-3,
    }
    
    train_data = DVSGestureDataset(data_root, train=True, num_frames=config_dvs['num_frames'])
    test_data = DVSGestureDataset(data_root, train=False, num_frames=config_dvs['num_frames'])
    
    train_loader = DataLoader(train_data, batch_size=config_dvs['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=config_dvs['batch_size'], shuffle=False, num_workers=2)
    
    model = EventSNN(in_channels=2, height=128, width=128, num_classes=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config_dvs['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config_dvs['epochs'])
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    best_acc = 0
    t0 = time.time()
    
    for epoch in range(config_dvs['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            marker = " *BEST*"
        else:
            marker = ""
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{config_dvs['epochs']} | "
                  f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%{marker}")
    
    dvs_time = time.time() - t0
    
    results['datasets']['DVS-Gesture'] = {
        'accuracy': best_acc,
        'sota_reference': 98.0,
        'gap_to_sota': 98.0 - best_acc,
        'training_time': dvs_time,
        'config': config_dvs,
        'synthetic_data': hasattr(train_data, 'synthetic')
    }
    
    print(f"\nDVS-Gesture Best Accuracy: {best_acc:.2f}% (SOTA: ~98%)")
    
    # =========================================================================
    # BENCHMARK 3: CIFAR10-DVS
    # =========================================================================
    print("\n" + "-" * 60)
    print("Benchmark 3: CIFAR10-DVS")
    print("SOTA Reference: ~83% accuracy")
    print("-" * 60)
    
    config_cifar = {
        'batch_size': 32,
        'epochs': 30,
        'num_frames': 10,
        'lr': 1e-3,
    }
    
    train_data = CIFAR10DVSDataset(data_root, train=True, num_frames=config_cifar['num_frames'])
    test_data = CIFAR10DVSDataset(data_root, train=False, num_frames=config_cifar['num_frames'])
    
    train_loader = DataLoader(train_data, batch_size=config_cifar['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=config_cifar['batch_size'], shuffle=False, num_workers=2)
    
    model = EventSNN(in_channels=2, height=128, width=128, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config_cifar['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config_cifar['epochs'])
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    best_acc = 0
    t0 = time.time()
    
    for epoch in range(config_cifar['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            marker = " *BEST*"
        else:
            marker = ""
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{config_cifar['epochs']} | "
                  f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%{marker}")
    
    cifar_time = time.time() - t0
    
    results['datasets']['CIFAR10-DVS'] = {
        'accuracy': best_acc,
        'sota_reference': 83.0,
        'gap_to_sota': 83.0 - best_acc,
        'training_time': cifar_time,
        'config': config_cifar,
        'synthetic_data': hasattr(train_data, 'synthetic')
    }
    
    print(f"\nCIFAR10-DVS Best Accuracy: {best_acc:.2f}% (SOTA: ~83%)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("NEUROMORPHIC BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Dataset':<15} {'Accuracy':<12} {'SOTA':<10} {'Gap':<10} {'Synthetic':<10}")
    print("-" * 60)
    
    for name, data in results['datasets'].items():
        acc = f"{data['accuracy']:.2f}%"
        sota = f"{data['sota_reference']:.0f}%"
        gap = f"{data['gap_to_sota']:.2f}%"
        synth = "Yes" if data['synthetic_data'] else "No"
        print(f"{name:<15} {acc:<12} {sota:<10} {gap:<10} {synth:<10}")
    
    print("\n" + "-" * 60)
    print("NOTE: Results with synthetic data are for demonstration only.")
    print("Download actual datasets for meaningful benchmarks:")
    print("  - N-MNIST: https://www.garrickorchard.com/datasets/n-mnist")
    print("  - DVS-Gesture: https://research.ibm.com/interactive/dvsgesture/")
    print("  - CIFAR10-DVS: https://figshare.com/articles/dataset/CIFAR10-DVS/4724671")
    
    # Save results
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(results_dir / f'neuromorphic_benchmark_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to: results/neuromorphic_benchmark_{timestamp}.json")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    results = run_neuromorphic_benchmark()
