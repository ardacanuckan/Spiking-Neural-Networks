# Spiking Neural Network Benchmark

A benchmarking framework for comparing spiking neural network (SNN) architectures on image classification tasks. Five models are implemented from scratch in PyTorch and evaluated under identical training conditions on MNIST, with infrastructure for CIFAR-10 and neuromorphic datasets (N-MNIST, DVS-Gesture, CIFAR10-DVS).

All models use surrogate gradient methods for training and are measured with metrics aligned to the [NeuroBench](https://arxiv.org/abs/2304.04640) framework: test accuracy, spike rate, activation sparsity, synaptic operations, and a theoretical energy ratio.

## Motivation

Most SNN papers report results using different training protocols, hyperparameters, and evaluation metrics, which makes direct comparison unreliable. This project puts five architectures—ranging from a vanilla LIF baseline to multi-compartment dendritic models—on equal footing. The goal is to quantify what each architectural choice actually buys you in terms of accuracy, sparsity, and energy.

## Architectures

| # | Model | Core Idea | Reference |
|---|-------|-----------|-----------|
| 1 | **Baseline LIF** | Standard leaky integrate-and-fire with fully connected layers | Textbook LIF |
| 2 | **DASNN** | Multi-compartment dendritic neurons with heterogeneous time constants and branch-level gating | Dendritic computation literature |
| 3 | **Spiking-KAN** | Kolmogorov-Arnold Network activations (learnable Chebyshev polynomials) driving LIF neurons | KAN (Liu et al., 2024) adapted to spikes |
| 4 | **NEXUS-SNN** | Chebyshev KAN + adaptive thresholds + temporal attention + residual connections | Combined approach |
| 5 | **APEX-SNN** | Time-to-first-spike coding, progressive sparsity enforcement, multi-scale ensemble readout | TTFS + structured pruning |

### Baseline LIF
Two fully connected layers (784→512→256→10) with LIF neurons (τ=2.0, threshold=1.0). Trained with ATan surrogate gradient. This is the control model.

### DASNN
Each neuron contains multiple dendritic branches with different membrane time constants (τ ∈ [1.5, 8.0]). Branches integrate inputs independently, then a learned weighting combines their contributions at the soma before the spike decision. This gives the neuron multi-timescale dynamics without extra layers.

### Spiking-KAN
Replaces fixed activation functions with learnable Chebyshev polynomial basis functions (degree 4). Each connection learns its own input-output mapping φ(x) = Σᵢ cᵢTᵢ(x), where Tᵢ are Chebyshev polynomials. The KAN output feeds into standard LIF neurons for spike generation.

### NEXUS-SNN
Stacks three ideas on top of each other: (1) Chebyshev KAN activations for richer representations, (2) adaptive firing thresholds that adjust based on recent activity, and (3) temporal attention that reweights spike contributions across time steps. Residual connections are added to ease gradient flow.

### APEX-SNN
Uses a TTFS-inspired coding scheme where early spikes carry more weight. A progressive sparsity mechanism gradually increases the fraction of pruned neurons during training (from 30% to 90%). The final classification uses an ensemble of readouts from multiple layers and time scales.

## Results

### MNIST (all models, identical protocol)

All models were trained on MNIST with batch size 128, Adam optimizer (lr=0.001 unless noted), and cross-entropy loss. Results are from a single run each—no cherry-picking, no hyperparameter sweeps.

| Model | Test Acc. | Spike Rate | Energy Eff.* | Params | Time Steps | Train Time |
|-------|-----------|------------|-------------|--------|------------|------------|
| Baseline LIF | 98.53% | 0.108 | 835.6 | 537K | 4 | — |
| DASNN | 98.54% | 0.101 | 887.2 | 668K | 4 | — |
| Spiking-KAN | 98.60% | 0.103 | 872.5 | 1.19M | 4 | — |
| NEXUS-SNN v2 | 99.09% | 0.123 | 747.6 | 671K | 6 | 406s |
| APEX-SNN | 99.21% | 0.139 | 664.9 | 1.60M | 2 | 571s |

*Energy efficiency = inverse of theoretical energy ratio (higher = more efficient relative to ANN baseline). Computed using 0.9 pJ/AC for spikes vs 4.6 pJ/MAC for ANNs at 45nm CMOS.

**Key observations:**
- All models exceed 98.5% on MNIST, which is expected—MNIST is nearly saturated.
- APEX-SNN reaches the highest accuracy (99.21%) but uses the most parameters (1.6M) and has the highest spike rate (0.139). It gets there in only 2 time steps.
- DASNN achieves the lowest spike rate (0.101) and best energy efficiency (887.2) among all models, while matching the baseline in accuracy. The dendritic branches appear to help with sparse coding.
- NEXUS-SNN v2 offers the best accuracy among models with <700K parameters (99.09%).
- Spiking-KAN's learnable activations give a modest accuracy gain over baseline (+0.07%) without increasing spike rate.

### Model comparison breakdown

**Who wins at what:**

| Metric | Winner | Value |
|--------|--------|-------|
| Highest accuracy | APEX-SNN | 99.21% |
| Lowest spike rate | DASNN | 0.101 |
| Best energy efficiency | DASNN | 887.2 |
| Fewest parameters | Baseline LIF | 537K |
| Fewest time steps | APEX-SNN | 2 |

### CIFAR-10 and neuromorphic benchmarks

Infrastructure for CIFAR-10 (SNN-VGG vs ANN comparison) and neuromorphic datasets (N-MNIST, DVS-Gesture, CIFAR10-DVS) is implemented but results are not yet finalized. The notebooks and experiment scripts are ready to run on a GPU.

For reference, current published SOTA on these benchmarks:

| Dataset | Published SOTA | Architecture | Year |
|---------|---------------|--------------|------|
| CIFAR-10 (SNN) | ~95–96% | Spikformer V2 | 2024 |
| N-MNIST | ~99% | Various | 2023 |
| DVS-Gesture | ~98% | Various | 2024 |
| CIFAR10-DVS | ~83% | Various | 2024 |

## Metrics

Following the NeuroBench standardized evaluation protocol:

- **Accuracy** — Standard top-1 classification accuracy on the test set.
- **Spike Rate** — Total spikes / (neurons × time steps). Lower means sparser activity.
- **Activation Sparsity** — Fraction of neurons that never fire. Range [0, 1].
- **Synaptic Operations (SynOps)** — Σ(spike_count × fan-out) per layer. SNNs use accumulate-only (AC) operations; ANNs use multiply-accumulate (MAC).
- **Energy Ratio** — (SynOps_SNN × E_AC) / (SynOps_ANN × E_MAC), where E_AC ≈ 0.9 pJ and E_MAC ≈ 4.6 pJ at 45nm.

## Installation

```bash
git clone https://github.com/ardacanuckan/Spiking-Neural-Networks.git
cd Spiking-Neural-Networks
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.8 and PyTorch ≥ 2.0.

## Usage

### Run all models on MNIST

```bash
python experiments/run_final_comparison.py
```

This trains Baseline LIF, DASNN, Spiking-KAN, and NEXUS-SNN sequentially and writes results to `experiments/results/` as JSON, with comparison plots saved to `experiments/figures/`.

### Run individual models

```bash
# APEX-SNN (trains for 35 epochs, ~10 min on GPU)
python experiments/run_apex_snn.py

# NEXUS-SNN v2
python experiments/run_nexus_v2.py

# Baseline vs DASNN
python experiments/run_comparison.py

# Baseline vs DASNN vs Spiking-KAN
python experiments/run_triple_comparison.py
```

### Notebooks (Colab-ready)

| Notebook | Description |
|----------|-------------|
| `notebooks/01_CIFAR10_SNN_Benchmark.ipynb` | CIFAR-10 SNN vs ANN comparison |
| `notebooks/02_Neuromorphic_Datasets.ipynb` | N-MNIST, DVS-Gesture, CIFAR10-DVS evaluation |
| `notebooks/03_All_Models_MNIST.ipynb` | All five models trained and compared on MNIST |

Each notebook is self-contained and can run on Google Colab with a T4 GPU.

## Project Structure

```
├── models/                     # Model implementations
│   ├── surrogate.py            # Surrogate gradient functions (ATan, SuperSpike, etc.)
│   ├── neurons.py              # Neuron models (LIF, AdaptiveLIF, TTFS, ParametricLIF)
│   ├── baseline_lif.py         # Baseline fully connected LIF network
│   ├── dasnn.py                # Dendritic Attention SNN
│   ├── spiking_kan.py          # Spiking Kolmogorov-Arnold Network
│   ├── nexus_snn.py            # NEXUS-SNN
│   └── apex_snn.py             # APEX-SNN
├── datasets/
│   └── loaders.py              # Data loaders (MNIST, CIFAR-10, neuromorphic)
├── src/                        # Modular components
│   ├── layers/                 # Spiking layers, attention mechanisms
│   ├── learning/               # Surrogate gradients, adaptive methods, loss functions
│   ├── models/                 # Base neuron classes, dendritic models
│   ├── training/               # Trainer class, callbacks (early stopping, checkpointing)
│   └── utils/                  # Visualization (spike rasters, membrane traces, training curves)
├── experiments/
│   ├── run_*.py                # Experiment scripts
│   ├── benchmark_*.py          # Benchmark scripts (CIFAR-10, neuromorphic)
│   ├── networks.py             # Network wrapper classes
│   ├── results/                # JSON output from experiments
│   └── figures/                # Generated comparison plots
├── notebooks/                  # Colab-ready Jupyter notebooks
├── requirements.txt
└── pyproject.toml
```

## Limitations

- **MNIST only.** Validated results are limited to MNIST. CIFAR-10 and neuromorphic dataset benchmarks are implemented but not yet trained to completion.
- **Single runs.** Each result is from one training run. No error bars, no repeated trials.
- **Theoretical energy only.** Energy estimates use the standard AC/MAC cost model. No measurements on actual neuromorphic hardware (Loihi, SpiNNaker, etc.).
- **No SynOps counting.** The current energy metric is a simplified ratio, not a proper per-layer SynOps count.
- **Synthetic neuromorphic data.** When real neuromorphic datasets (N-MNIST, DVS-Gesture) are unavailable, the loaders fall back to synthetic approximations.

## Planned Work

- [ ] Complete CIFAR-10 and CIFAR-100 benchmarks with full training
- [ ] Run on real neuromorphic datasets (requires manual download)
- [ ] Implement proper per-layer SynOps counting
- [ ] Add error bars from multiple runs
- [ ] Compare against SpikingJelly and snnTorch baselines
- [ ] Test on neuromorphic hardware

## References

- Yik, J. et al. (2025). NeuroBench: Advancing Neuromorphic Computing through Collaborative, Fair and Representative Benchmarking. *Nature Communications*.
- Liu, Z. et al. (2024). KAN: Kolmogorov-Arnold Networks. *arXiv:2404.19756*.
- Gerstner, W. & Kistler, W. (2002). *Spiking Neuron Models*. Cambridge University Press.
- Neftci, E. O. et al. (2019). Surrogate Gradient Learning in Spiking Neural Networks. *IEEE Signal Processing Magazine*.
- Fang, W. et al. (2021). SpikingJelly. GitHub repository.

## License

MIT — see [LICENSE](LICENSE).
