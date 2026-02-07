"""
Visualization utilities for Spiking Neural Networks.

Provides functions for:
- Spike raster plots
- Membrane potential traces
- Attention weight visualization
- Training curve plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union
from pathlib import Path


def plot_spike_raster(
    spikes: Union[torch.Tensor, np.ndarray],
    ax: Optional[plt.Axes] = None,
    time_range: Optional[Tuple[int, int]] = None,
    neuron_range: Optional[Tuple[int, int]] = None,
    title: str = "Spike Raster Plot",
    xlabel: str = "Time Step",
    ylabel: str = "Neuron Index",
    marker_size: float = 0.5,
    color: str = 'black',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a spike raster plot.
    
    Args:
        spikes: Spike tensor [time, neurons] or [time, batch, neurons]
        ax: Matplotlib axes (creates new figure if None)
        time_range: Optional tuple (start, end) for time range
        neuron_range: Optional tuple (start, end) for neuron range
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        marker_size: Size of spike markers
        color: Color of spike markers
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()
    
    # Handle batch dimension
    if spikes.ndim == 3:
        spikes = spikes[:, 0, :]  # Take first batch
    
    # Apply ranges
    if time_range is not None:
        spikes = spikes[time_range[0]:time_range[1]]
    if neuron_range is not None:
        spikes = spikes[:, neuron_range[0]:neuron_range[1]]
    
    # Find spike times and neuron indices
    spike_times, neuron_ids = np.where(spikes > 0)
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Plot spikes
    ax.scatter(spike_times, neuron_ids, s=marker_size, c=color, marker='|')
    
    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set limits
    ax.set_xlim(-0.5, spikes.shape[0] - 0.5)
    ax.set_ylim(-0.5, spikes.shape[1] - 0.5)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_membrane_potential(
    membrane: Union[torch.Tensor, np.ndarray],
    threshold: float = 1.0,
    spikes: Optional[Union[torch.Tensor, np.ndarray]] = None,
    neuron_indices: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Membrane Potential Trace",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot membrane potential traces over time.
    
    Args:
        membrane: Membrane potential tensor [time, neurons]
        threshold: Firing threshold for reference line
        spikes: Optional spike tensor for marking spike times
        neuron_indices: Which neurons to plot (default: first 5)
        ax: Matplotlib axes
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if isinstance(membrane, torch.Tensor):
        membrane = membrane.detach().cpu().numpy()
    if spikes is not None and isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()
    
    # Handle batch dimension
    if membrane.ndim == 3:
        membrane = membrane[:, 0, :]
        if spikes is not None:
            spikes = spikes[:, 0, :]
    
    # Select neurons to plot
    if neuron_indices is None:
        neuron_indices = list(range(min(5, membrane.shape[1])))
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    time_steps = np.arange(membrane.shape[0])
    
    # Plot membrane potentials
    for i, neuron_idx in enumerate(neuron_indices):
        ax.plot(time_steps, membrane[:, neuron_idx], 
                label=f'Neuron {neuron_idx}', alpha=0.8)
        
        # Mark spikes if provided
        if spikes is not None:
            spike_times = np.where(spikes[:, neuron_idx] > 0)[0]
            ax.scatter(spike_times, membrane[spike_times, neuron_idx],
                      marker='*', s=100, zorder=5)
    
    # Threshold line
    ax.axhline(y=threshold, color='r', linestyle='--', 
               label=f'Threshold ({threshold})', alpha=0.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Membrane Potential')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attention_weights(
    attention: Union[torch.Tensor, np.ndarray],
    time_step: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Dendritic Attention Weights",
    cmap: str = 'viridis',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention weights from dendritic attention layer.
    
    Args:
        attention: Attention weights [time, batch, heads] or [batch, heads]
        time_step: Which time step to visualize (if temporal)
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()
    
    # Select time step if needed
    if attention.ndim == 3:
        if time_step is None:
            time_step = -1  # Last time step
        attention = attention[time_step]
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Plot heatmap
    im = ax.imshow(attention, aspect='auto', cmap=cmap)
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    ax.set_xlabel('Attention Head (Dendrite)')
    ax.set_ylabel('Sample')
    ax.set_title(title)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    history: dict,
    metrics: List[str] = ['train_loss', 'val_loss', 'train_acc', 'val_acc'],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with training history
        metrics: Which metrics to plot
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Determine number of subplots
    loss_metrics = [m for m in metrics if 'loss' in m]
    acc_metrics = [m for m in metrics if 'acc' in m]
    
    n_plots = sum([len(loss_metrics) > 0, len(acc_metrics) > 0])
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Loss subplot
    if loss_metrics:
        ax = axes[plot_idx]
        for metric in loss_metrics:
            if metric in history:
                label = metric.replace('_', ' ').title()
                ax.plot(history[metric], label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Accuracy subplot
    if acc_metrics:
        ax = axes[plot_idx]
        for metric in acc_metrics:
            if metric in history:
                label = metric.replace('_', ' ').title()
                ax.plot(history[metric], label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_firing_rate_distribution(
    spikes: Union[torch.Tensor, np.ndarray],
    ax: Optional[plt.Axes] = None,
    bins: int = 50,
    title: str = "Firing Rate Distribution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of firing rates across neurons.
    
    Args:
        spikes: Spike tensor [time, neurons] or [time, batch, neurons]
        ax: Matplotlib axes
        bins: Number of histogram bins
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()
    
    # Calculate firing rates
    if spikes.ndim == 3:
        firing_rates = spikes.mean(axis=(0, 1))  # Average over time and batch
    else:
        firing_rates = spikes.mean(axis=0)  # Average over time
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Plot histogram
    ax.hist(firing_rates, bins=bins, edgecolor='black', alpha=0.7)
    ax.axvline(x=firing_rates.mean(), color='r', linestyle='--',
               label=f'Mean: {firing_rates.mean():.3f}')
    
    ax.set_xlabel('Firing Rate')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
