import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path

try:
    from spikingjelly.datasets import n_mnist, dvs128_gesture, cifar10_dvs
    SPIKINGJELLY_AVAILABLE = True
except ImportError:
    SPIKINGJELLY_AVAILABLE = False


def get_mnist_loaders(batch_size=128, data_dir='./data', num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def get_cifar10_loaders(batch_size=128, data_dir='./data', num_workers=2, augment=True):
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


class SyntheticNMNIST(Dataset):
    def __init__(self, n_samples, num_frames=10):
        self.n_samples = n_samples
        self.num_frames = num_frames
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        label = idx % 10
        frames = torch.rand(self.num_frames, 2, 34, 34) * 0.3
        frames[:, :, 10:24, 10:24] += label * 0.05
        return frames, label


class SyntheticDVSGesture(Dataset):
    def __init__(self, n_samples, num_frames=16):
        self.n_samples = n_samples
        self.num_frames = num_frames
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        label = idx % 11
        frames = torch.zeros(self.num_frames, 2, 128, 128)
        for t in range(self.num_frames):
            cx = 64 + int(30 * np.sin(2 * np.pi * (t + label) / self.num_frames))
            cy = 64 + int(30 * np.cos(2 * np.pi * (t + label * 2) / self.num_frames))
            frames[t, t % 2, max(0,cy-10):min(128,cy+10), max(0,cx-10):min(128,cx+10)] = 1.0
        return frames, label


class SyntheticCIFAR10DVS(Dataset):
    def __init__(self, n_samples, num_frames=10):
        self.n_samples = n_samples
        self.num_frames = num_frames
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        label = idx % 10
        frames = torch.rand(self.num_frames, 2, 128, 128) * 0.3
        frames[:, :, ::4, ::4] += label * 0.05
        return frames.clamp(0, 1), label


def get_nmnist_loaders(batch_size=64, data_dir='./data', num_frames=10, num_workers=2):
    if SPIKINGJELLY_AVAILABLE:
        try:
            train_data = n_mnist.NMNIST(
                root=f'{data_dir}/NMNIST',
                train=True,
                data_type='frame',
                frames_number=num_frames,
                split_by='number'
            )
            test_data = n_mnist.NMNIST(
                root=f'{data_dir}/NMNIST',
                train=False,
                data_type='frame',
                frames_number=num_frames,
                split_by='number'
            )
        except Exception:
            print("N-MNIST download failed, using synthetic data")
            train_data = SyntheticNMNIST(60000, num_frames)
            test_data = SyntheticNMNIST(10000, num_frames)
    else:
        print("SpikingJelly not available, using synthetic N-MNIST data")
        train_data = SyntheticNMNIST(60000, num_frames)
        test_data = SyntheticNMNIST(10000, num_frames)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def get_dvs_gesture_loaders(batch_size=16, data_dir='./data', num_frames=16, num_workers=2):
    if SPIKINGJELLY_AVAILABLE:
        try:
            train_data = dvs128_gesture.DVS128Gesture(
                root=f'{data_dir}/DVSGesture',
                train=True,
                data_type='frame',
                frames_number=num_frames,
                split_by='number'
            )
            test_data = dvs128_gesture.DVS128Gesture(
                root=f'{data_dir}/DVSGesture',
                train=False,
                data_type='frame',
                frames_number=num_frames,
                split_by='number'
            )
        except Exception:
            print("DVS-Gesture download failed, using synthetic data")
            train_data = SyntheticDVSGesture(1342, num_frames)
            test_data = SyntheticDVSGesture(288, num_frames)
    else:
        print("SpikingJelly not available, using synthetic DVS-Gesture data")
        train_data = SyntheticDVSGesture(1342, num_frames)
        test_data = SyntheticDVSGesture(288, num_frames)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def get_cifar10_dvs_loaders(batch_size=32, data_dir='./data', num_frames=10, num_workers=2):
    if SPIKINGJELLY_AVAILABLE:
        try:
            full_data = cifar10_dvs.CIFAR10DVS(
                root=f'{data_dir}/CIFAR10DVS',
                data_type='frame',
                frames_number=num_frames,
                split_by='number'
            )
            n_train = int(0.9 * len(full_data))
            n_test = len(full_data) - n_train
            train_data, test_data = torch.utils.data.random_split(full_data, [n_train, n_test])
        except Exception:
            print("CIFAR10-DVS download failed, using synthetic data")
            train_data = SyntheticCIFAR10DVS(9000, num_frames)
            test_data = SyntheticCIFAR10DVS(1000, num_frames)
    else:
        print("SpikingJelly not available, using synthetic CIFAR10-DVS data")
        train_data = SyntheticCIFAR10DVS(9000, num_frames)
        test_data = SyntheticCIFAR10DVS(1000, num_frames)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader
