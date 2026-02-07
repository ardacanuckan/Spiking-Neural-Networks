#!/usr/bin/env python3
"""
MNIST Training Script for DASNN

This script trains the Dendritic Attention Spiking Neural Network (DASNN)
on the MNIST dataset to demonstrate our novel contributions:

1. Multi-compartment dendritic neurons
2. Dendritic attention mechanism  
3. Heterogeneous time constants
4. Adaptive surrogate gradients

Usage:
    python train_mnist.py --model dasnn --epochs 100 --time_steps 4
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from networks import DASNNClassifier, ConvDASNN, get_model


class ATanSurrogate(torch.autograd.Function):
    """Arctangent surrogate gradient."""
    
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        import math
        grad = 1.0 / (math.pi * (1 + (math.pi * ctx.alpha * x) ** 2))
        return grad * grad_output, None


def atan_surrogate(x, alpha=2.0):
    """Surrogate gradient function."""
    return ATanSurrogate.apply(x, alpha)


class SNNWrapper(nn.Module):
    """Wrapper to make model compatible with standard training."""
    
    def __init__(self, model, time_steps=4):
        super().__init__()
        self.model = model
        self.time_steps = time_steps
    
    def forward(self, x):
        # Reset states
        self.model.reset_state()
        
        # Accumulate outputs over time steps
        outputs = []
        for t in range(self.time_steps):
            out = self.model(x, atan_surrogate)
            outputs.append(out)
        
        # Average over time (rate coding)
        return torch.stack(outputs).mean(dim=0)


def get_data_loaders(batch_size=128, data_dir='./data'):
    """Get MNIST data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}: '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(train_loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(test_loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train DASNN on MNIST')
    parser.add_argument('--model', type=str, default='dasnn',
                        choices=['dasnn', 'conv_dasnn'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--time_steps', type=int, default=4,
                        help='Number of time steps for SNN')
    parser.add_argument('--hidden_sizes', type=int, nargs='+',
                        default=[512, 256],
                        help='Hidden layer sizes')
    parser.add_argument('--num_branches', type=int, default=4,
                        help='Number of dendritic branches')
    parser.add_argument('--tau_min', type=float, default=2.0,
                        help='Minimum time constant')
    parser.add_argument('--tau_max', type=float, default=16.0,
                        help='Maximum time constant')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_attention', action='store_true',
                        help='Disable dendritic attention')
    parser.add_argument('--save_dir', type=str, default='../results',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data
    train_loader, test_loader = get_data_loaders(args.batch_size)
    
    # Model
    if args.model == 'dasnn':
        base_model = DASNNClassifier(
            input_size=784,
            hidden_sizes=args.hidden_sizes,
            num_classes=10,
            num_branches=args.num_branches,
            tau_range=(args.tau_min, args.tau_max),
            use_dendritic_attention=not args.no_attention,
            dropout=args.dropout
        )
    else:  # conv_dasnn
        base_model = ConvDASNN(
            in_channels=1,
            num_classes=10,
            channels=[32, 64, 128],
            tau=args.tau_min
        )
    
    model = SNNWrapper(base_model, time_steps=args.time_steps).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.model}')
    print(f'Parameters: {num_params:,}')
    print(f'Time steps: {args.time_steps}')
    print(f'Dendritic attention: {not args.no_attention}')
    print(f'Tau range: [{args.tau_min}, {args.tau_max}]')
    print(f'Hidden sizes: {args.hidden_sizes}')
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Results tracking
    results = {
        'config': vars(args),
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'best_acc': 0.0
    }
    
    # Training loop
    print('\n' + '='*60)
    print('Starting training...')
    print('='*60)
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record results
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        
        if test_acc > results['best_acc']:
            results['best_acc'] = test_acc
            # Save best model
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        print(f'Best Acc: {results["best_acc"]:.2f}%')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = save_dir / f'results_{args.model}_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n' + '='*60)
    print('Training complete!')
    print(f'Best test accuracy: {results["best_acc"]:.2f}%')
    print(f'Results saved to: {results_file}')
    print('='*60)


if __name__ == '__main__':
    main()
