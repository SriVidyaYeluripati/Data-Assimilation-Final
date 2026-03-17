# Data-Assimilation

A reproducible Python framework for data assimilation experiments on the Lorenz-63 system, comparing neural network architectures (MLP, GRU, LSTM) as learned approximators of the 3D-Var analysis functional.

## Overview

This repository implements the codebase for the Masterpraktikum report:

> **Approximating the Variational Data Assimilation Functional via Neural Networks —
> A Pilot Study on the Lorenz-63 System under Varying Observation Maps and Noise Levels**

The framework trains neural networks to approximate the 3D-Var analysis functional $\Phi$ in a self-supervised manner — minimising the variational cost directly, without access to ground-truth analysis states. Three architectures are compared (MLP, GRU, LSTM) across multiple observation maps and noise levels on the chaotic Lorenz-63 system.

**Key features:**

- Generation of synthetic trajectories from the Lorenz-63 ODE system (1500 total: 1000 train / 500 test).
- Three observation maps of varying complexity: $h(x) = x_1$ (partial linear), $h(x) = (x_1, x_2)^\top$ (two-component linear), and $h(x) = x_1^2$ (nonlinear).
- Four observation noise levels: $\sigma_\text{obs} \in \{0.05, 0.10, 0.50, 1.00\}$.
- Two background conditioning strategies: **FixedMean** (static ensemble mean throughout training) and **Resample** (background drawn stochastically from $\mathcal{N}(\bar{x}_\text{ens}, B_\text{ens})$ per sample).
- Self-supervised training: the network learns by minimising the 3D-Var cost $J(x)$ evaluated at its output — no ground-truth analysis labels required.
- Full experiment orchestrator: training, evaluation, diagnostics, and model artefact saving.
- Visual diagnostics: trajectory reconstructions (truth vs background vs analysis), error evolution plots, attractor geometry panels, and failure case visualisations.
- Strict train/test separation to avoid data leakage and ensure fair comparisons across architectures.

---

## Repository Structure

```
data/
  └── ...                # synthetic trajectories, observation sequences, background statistics
results/
  └── ...                # saved model weights, metrics, figures
src/
  ├── simulator.py       # Lorenz-63 integrator and trajectory generator
  ├── dataset.py         # AssimilationDataset, observation map implementations
  ├── models.py          # MLP, GRU, and LSTM architecture definitions
  ├── loss.py            # Self-supervised 3D-Var variational loss (VarLoss)
  ├── trainer.py         # Experiment orchestrator: training loops, evaluation
  ├── evaluation.py      # Post-assimilation metrics (RMSE, Hausdorff, lobe occupancy)
  ├── visualize.py       # Reconstruction panels, error plots, attractor geometry figures
  └── utils.py           # Save/load utilities, logging, config helpers
README.md
```

---

## Getting Started

### Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib

Install dependencies via:

```bash
pip install -r requirements.txt
```

### Running an Experiment

**1. Generate synthetic data:**

```bash
python src/simulator.py --n_traj 1500 --train 1000 --test 500 --obs_mode xy
```

This integrates 1500 Lorenz-63 trajectories (1000 training, 500 test), applies the chosen observation map, and saves the dataset to `data/`.

**2. Train a model:**

```bash
python src/trainer.py --model gru --obs_mode xy --strategy resample --save_dir results/gru_xy_resample
```

This trains the GRU architecture under the Resample background conditioning strategy, saves model weights, and produces evaluation metrics.

**3. Evaluate and visualise:**

```bash
python src/visualize.py --result_dir results/gru_xy_resample
```

This produces trajectory reconstruction panels, error evolution plots, and attractor geometry figures.

Repeat for other combinations:
- Architectures: `mlp`, `gru`, `lstm`
- Observation maps: `x`, `xy`, `x2`
- Background strategies: `resample`, `fixedmean`, `baseline`

---

## Experiment Design

| Parameter | Values |
|---|---|
| Trajectories (train / test) | 1000 / 500 |
| Time steps per trajectory | 200 |
| Integration step $\Delta t$ | 0.01 |
| Observation maps | $x_1$, $(x_1, x_2)^\top$, $x_1^2$ |
| Noise levels $\sigma_\text{obs}$ | 0.05, 0.10, 0.50, 1.00 |
| Sequence window $L$ | 5 |
| Architectures | MLP, GRU, LSTM |
| Hidden dimension | 64 (32 for Baseline MLP) |
| Batch size | 256 |
| Training epochs | 30 |
| Optimizer | Adam, lr = 1e-3 |

The **Baseline MLP** uses a smaller hidden dimension of 32 and receives no background information, minimising only the observation cost $\mathcal{J}_o$. It serves as a diagnostic lower bound to quantify the benefit of incorporating background prior information.

---

## Evaluation Metrics

- **RMSE** — primary accuracy metric against held-out test trajectories
- **Root Median Squared Error (RMdSE)** — robust alternative when divergent runs inflate the mean
- **Normalised Hausdorff distance** $\tilde{H}_\text{global}$ — geometric fidelity of the reconstructed attractor
- **Lobe occupancy discrepancy** $\Delta_\text{lobe}$ — global topology preservation (correct separatrix crossing behaviour)

A run is classified as **diverged** if post-assimilation RMSE exceeds $10\times$ the median for that configuration or if $\tilde{H}_\text{global} > 10$.

---

## Project Scope

This repository serves as the full codebase for the AI-Var Masterpraktikum report. Its goals are:

- Provide a structured, reproducible pipeline for neural network-based variational data assimilation.
- Compare MLP, GRU, and LSTM architectures under identical experimental conditions (same data, same loss, same observation maps).
- Investigate the effect of background conditioning strategy (FixedMean vs Resample) on training stability and generalisation.
- Emphasise geometric diagnostics beyond aggregate RMSE to assess attractor fidelity.
