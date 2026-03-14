# SFCM: Surrogate-Feedback Co-evolutionary Metaheuristic

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

Official implementation of the paper:

> **SFCM: A Data-Driven Co-evolutionary Framework for Knowledge-Based Expensive Optimization**  
> Seyed Hossein Ahmadpanah, Department of Computer Engineering, Islamic Azad University, Tehran, Iran.

---

## Reproducibility Statement

This repository contains **complete source code, Docker containers, and hyperparameter configuration files** to reproduce every table and figure in the paper. All hyperparameter values match **Appendix C Table C1** (the authoritative source). See [Hyperparameter Notes](#hyperparameter-notes) below for the corrections applied in the revised manuscript.

---

## Repository Structure

```
sfcm/
├── sfcm/                    # Core SFCM implementation
│   ├── core.py              # Algorithm 1 & 2 — main optimisation loop
│   ├── surrogate.py         # EnsembleMLP + SurrogatePopulation (Appendix B)
│   ├── operators.py         # All operators: Eqs.(4–9), LHS, DE/rand/1/bin
│   └── __init__.py
│
├── benchmarks/
│   └── functions.py         # All 12 synthetic functions (§4.2.1)
│
├── problems/
│   └── real_world.py        # P1 ResNet-18 HPO, P2 Truss, P3 Container (§4.2.2)
│
├── baselines/
│   └── registry.py          # GA, PSO, DE, HHO, MPA, SMA, SA-DE,
│                            #   GP-EI, SMAC3, CMA-ES (§4.3)
│
├── experiments/
│   └── run_experiments.py   # Reproduces Tables 2–5 and Figs 2–7
│
├── scripts/
│   └── plot_figures.py      # Figure generation from saved CSV results
│
├── configs/
│   └── sfcm_default.yaml    # ← Single source of truth for all hyperparameters
│
├── tests/
│   └── test_sfcm.py         # Unit + integration tests (includes parameter guards)
│
├── docker/
│   └── Dockerfile           # CPU and GPU targets
│
└── requirements.txt         # Pinned dependencies
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/ahmadpanah/sfcm.git
cd sfcm
pip install -r requirements.txt
```

### 2. Run unit tests

```bash
pytest tests/ -v
```

The test suite includes **hyperparameter consistency guards** that will fail if the corrected values (κ=1.5, β=0.8, τ₀=0.3) are accidentally reverted.

### 3. Quick optimisation example

```python
from sfcm import SFCM
import numpy as np

def rastrigin(x):
    d = len(x)
    return 10*d + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)

bounds = np.array([[-5.12, 5.12]] * 10)   # 10D Rastrigin
opt = SFCM(rastrigin, bounds, budget=10_000, seed=0)
best_x, best_y = opt.run()
print(f"Best value: {best_y:.6f}")
```

### 4. Reproduce all paper results

```bash
# Full 30-run experiment (requires GPU for P1; use --simulate for CPU)
python experiments/run_experiments.py --all --n_runs 30 --simulate

# Individual components
python experiments/run_experiments.py --bench Rastrigin --d 10 30 50 --n_runs 30
python experiments/run_experiments.py --ablation --d 30 --n_runs 30
python experiments/run_experiments.py --realworld --simulate

# Generate all figures from results
python scripts/plot_figures.py --demo
```

Results are saved to `results/tables/` (CSV) and `results/figures/` (PDF).

### 5. Docker (exact environment)

```bash
# CPU — unit tests only
docker build -f docker/Dockerfile --target cpu -t sfcm:cpu .
docker run --rm sfcm:cpu pytest tests/ -v

# GPU — full experiments (requires NVIDIA Docker runtime)
docker build -f docker/Dockerfile --target gpu -t sfcm:gpu .
docker run --rm --gpus all sfcm:gpu \
    python experiments/run_experiments.py --all --n_runs 30
```

---

## Hyperparameter Notes

The revised manuscript corrects three transcription errors in §4.4 of the original draft. **All experiments were run with the values in Appendix C Table C1**, which is the authoritative source. The config file `configs/sfcm_default.yaml` and the code in `sfcm/core.py` use the correct values:

| Parameter | §4.4 draft (incorrect) | Appendix C / Code (correct) | Role |
|-----------|----------------------|----------------------------|------|
| κ (UCB weight) | 0.5 | **1.5** | Controls exploration bonus in noise-aware selection (Eq. 6) |
| β (expl. boost) | 1.5 | **0.8** | Misprediction amplification factor (Eq. 9) |
| τ₀ (init. threshold) | 0.4 | **0.3** | Initial trust threshold for resource allocation (Eq. 4) |

The unit test `TestHyperparameterConsistency` in `tests/test_sfcm.py` permanently guards against re-introduction of these errors.

---

## Algorithm Overview

SFCM maintains two co-evolving populations:

```
P^X_t   (solution population, size N=30)  — navigates decision space Ω
P^Θ_t   (surrogate population, size M=10) — adapts the predictive model
```

At each generation:

1. **Resource allocation** (Eq. 4–5): trust score decides whether to query `f(x)` (expensive) or use surrogate `ŷ(x)` (cheap).
2. **Misprediction density** (Eq. 8): kernel smoother identifies regions of high past surrogate error.
3. **Guided offspring** (Eq. 9): DE/rand/1/bin mutation amplified toward high-error regions.
4. **Noise-aware selection** (Eq. 6–7): UCB scores replace standard fitness; robust replacement tolerance δₜ decays as budget is consumed.
5. **Surrogate co-evolution** (Eq. 13): meta-fitness `F_t(θ) = λ·Lpred + (1-λ)·Lopt` drives DE/rand/1/bin variation in Θ-space, where θ encodes MLP hidden widths, dropout rate, and learning rate.

---

## Ablation Variants

| Variant | Disabled mechanism |
|---------|-------------------|
| `SFCM-noDP` | Dual-population co-evolution; reverts to single static surrogate |
| `SFCM-noNS` | Noise-aware UCB selection; uses raw fitness comparison |
| `SFCM-noME` | Misprediction-guided exploration; uses unscaled DE perturbation |
| `Base (DE)` | All surrogate components removed |

---

## Real-World Problem Definitions

| Problem | d | Variables | Objective | Budget |
|---------|---|-----------|-----------|--------|
| P1 ResNet-18 HPO | 8 | lr, weight_decay, batch_size, dropout, momentum, epochs, scheduler_step, γ | 1 − val_accuracy (CIFAR-10) | 100 |
| P2 10-Bar Truss | 10 | Cross-sectional areas Aᵢ ∈ [0.1, 35.0] cm² | Total weight (kN), s.t. stress ≤25 kN/cm², disp ≤2 cm | 500 |
| P3 Container Placement | 15 | Integer node assignments ∈ {0,…,29}^15 | Mean RTT latency (ms), s.t. CPU ≤80%, MEM ≤90%, hops ≤2 | 1000 |

Full formal definitions: `problems/real_world.py`

---

## License

MIT License — see [LICENSE](LICENSE).
