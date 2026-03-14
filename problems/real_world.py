"""
problems/real_world.py
======================
Formal definitions of the three real-world expensive optimisation problems
described in §4.2.2 (revised).

P1 : ResNet-18 Hyperparameter Optimisation on CIFAR-10
P2 : 10-Bar Planar Truss Structural Design
P3 : Container Placement Latency Minimisation (30-node cluster)

Each problem exposes:
    bounds  : np.ndarray  shape (d, 2)
    budget  : int          max true evaluations B
    evaluate(x) -> float   the expensive objective (to be minimised)

Notes
-----
P1 requires PyTorch + torchvision.  To avoid GPU dependency during unit
tests, a lightweight surrogate simulator is provided (simulate=True).

P2 uses a pure-Python finite-element solver; no external dependency.

P3 uses a pure-Python discrete-event simulator.
"""

from __future__ import annotations

import numpy as np
import time
from typing import Optional


# ============================================================================
# P1 — ResNet-18 Hyperparameter Optimisation
# ============================================================================

# Decision variables and bounds (§4.2.2, revised)
#   0  learning rate η         [1e-5, 1e-1]   log-uniform → encoded as log10
#   1  weight decay λ_w        [1e-6, 1e-2]   log-uniform → encoded as log10
#   2  batch size b            [0, 3] → {32, 64, 128, 256}  (integer index)
#   3  dropout rate p          [0.0, 0.5]
#   4  momentum m              [0.8, 0.999]
#   5  training epochs E       [20, 200]       integer
#   6  LR scheduler step s     [5, 50]         integer
#   7  LR scheduler γ          [0.1, 0.9]

P1_BOUNDS = np.array([
    [-5.0, -1.0],   # log10(η)
    [-6.0, -2.0],   # log10(λ_w)
    [0.0,  3.0],    # batch_size index (continuous; rounded to int)
    [0.0,  0.5],    # dropout
    [0.8,  0.999],  # momentum
    [20.0, 200.0],  # epochs
    [5.0,  50.0],   # scheduler step
    [0.1,  0.9],    # scheduler γ
])
P1_BATCH_SIZES = [32, 64, 128, 256]
P1_BUDGET = 100


class ResNet18HPO:
    """
    P1: ResNet-18 HPO on CIFAR-10.

    Parameters
    ----------
    simulate : bool
        If True, uses a fast polynomial surrogate instead of real training.
        Set simulate=False for actual GPU experiments.
    data_dir : str
        Path to CIFAR-10 data root (only used when simulate=False).
    device : str
        'cuda' or 'cpu' (only used when simulate=False).
    """

    bounds = P1_BOUNDS
    budget = P1_BUDGET
    d = 8

    def __init__(
        self,
        simulate: bool = True,
        data_dir: str = "./data",
        device: str = "cuda",
    ):
        self.simulate = simulate
        self.data_dir = data_dir
        self.device = device

    def evaluate(self, x: np.ndarray) -> float:
        """
        Returns  1 − validation_accuracy  (minimisation convention).

        x[0] : log10(learning_rate)
        x[1] : log10(weight_decay)
        x[2] : batch_size_index  → {32,64,128,256}
        x[3] : dropout
        x[4] : momentum
        x[5] : epochs            (rounded to int)
        x[6] : scheduler_step    (rounded to int)
        x[7] : scheduler_gamma
        """
        hp = self._decode(x)

        if self.simulate:
            return self._simulate(hp)
        else:
            return self._train_real(hp)

    # ------------------------------------------------------------------
    @staticmethod
    def _decode(x: np.ndarray) -> dict:
        return dict(
            lr=float(10 ** x[0]),
            weight_decay=float(10 ** x[1]),
            batch_size=P1_BATCH_SIZES[int(np.clip(round(x[2]), 0, 3))],
            dropout=float(np.clip(x[3], 0.0, 0.5)),
            momentum=float(np.clip(x[4], 0.8, 0.999)),
            epochs=int(np.clip(round(x[5]), 20, 200)),
            scheduler_step=int(np.clip(round(x[6]), 5, 50)),
            scheduler_gamma=float(np.clip(x[7], 0.1, 0.9)),
        )

    @staticmethod
    def _simulate(hp: dict) -> float:
        """
        Lightweight polynomial surrogate for CI / unit testing.
        Mimics the rough accuracy landscape without GPU.
        Best acc ≈ 93.5% at known-good hyperparameters.
        Returns 1 - accuracy (minimisation).
        """
        # Centred log-lr (optimum around -2.0)
        lr_term = (np.log10(hp["lr"]) + 2.0) ** 2

        # Regularisation penalty
        wd_term = (np.log10(hp["weight_decay"]) + 4.0) ** 2 * 0.3

        # Epoch benefit (more epochs better up to ~100)
        epoch_term = max(0.0, (100 - hp["epochs"]) / 100.0) * 2.0

        # Dropout (moderate best)
        drop_term = (hp["dropout"] - 0.2) ** 2 * 5.0

        # Momentum (high is good)
        mom_term = (0.999 - hp["momentum"]) * 2.0

        noise = np.random.default_rng().normal(0, 0.005)
        raw_error = 0.065 + 0.01 * (lr_term + wd_term + epoch_term + drop_term + mom_term) + noise
        return float(np.clip(raw_error, 0.01, 0.99))

    def _train_real(self, hp: dict) -> float:
        """Full ResNet-18 training on CIFAR-10.  Requires GPU environment."""
        try:
            import torch
            import torch.nn as nn
            import torchvision
            import torchvision.transforms as transforms
        except ImportError:
            raise ImportError("PyTorch and torchvision required for real P1 evaluation.")

        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform_test
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=hp["batch_size"], shuffle=True, num_workers=4
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=256, shuffle=False, num_workers=4
        )

        model = torchvision.models.resnet18(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model = model.to(self.device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hp["lr"],
            momentum=hp["momentum"],
            weight_decay=hp["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=hp["scheduler_step"], gamma=hp["scheduler_gamma"]
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(hp["epochs"]):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                criterion(model(inputs), targets).backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                _, predicted = model(inputs).max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_acc = correct / total
        return float(1.0 - val_acc)   # minimise → return error rate


# ============================================================================
# P2 — 10-Bar Planar Truss Structural Design
# ============================================================================

# Standard benchmark: Schmit (1960), widely reproduced in the literature.
# d = 10 cross-sectional areas A_i ∈ [0.1, 35.0] cm²
# Objective: minimise total weight W = ρ Σ Aᵢ Lᵢ
# Subject to:  |σᵢ| ≤ 25 kN/cm²   (stress)
#              |δⱼ| ≤ 2.0 cm       (nodal displacement)

P2_BOUNDS = np.array([[0.1, 35.0]] * 10)
P2_BUDGET = 500

# Geometry (m): 10 nodes, 10 bars, horizontal span L=9.144m, height H=9.144m
_L = 9.144         # m → convert to cm: L_cm = _L * 100
_H = 9.144         # m
# Node coordinates (x, y) in metres
_NODES = np.array([
    [2 * _L, _H],   # 0
    [2 * _L, 0.0],  # 1
    [_L,     _H],   # 2
    [_L,     0.0],  # 3
    [0.0,    _H],   # 4
    [0.0,    0.0],  # 5
])
# Bar connectivity (node_i, node_j)
_BARS = [
    (0, 2), (2, 4), (1, 3), (3, 5),   # horizontal/vertical chords
    (0, 3), (2, 5), (0, 1), (2, 3),   # diagonals
    (4, 5), (2, 1),                    # remaining
]
_RHO  = 0.1    # density kN/cm³ (aluminium equivalent)
_E    = 6.9e7  # Young's modulus kN/cm²
_P    = 100.0  # external load kN (applied downward at nodes 1, 3)


class TrussDesign:
    """
    P2: 10-bar planar truss weight minimisation.

    evaluate(x) -> weight [kN]   subject to stress and displacement constraints.
    Constraint violations are handled via exterior penalty.
    """

    bounds = P2_BOUNDS
    budget = P2_BUDGET
    d = 10

    def evaluate(self, x: np.ndarray) -> float:
        A = np.clip(x, 0.1, 35.0)          # cm²
        return self._fem_objective(A)

    # ------------------------------------------------------------------
    @staticmethod
    def _fem_objective(A: np.ndarray) -> float:
        """
        Simple 2D truss FEM.
        Returns weight + penalty for constraint violations.
        """
        n_bars = 10
        n_nodes = 6
        dof = 2 * n_nodes   # 12 DOFs

        # Bar lengths [cm]
        L_bars = np.array([
            np.linalg.norm(_NODES[b] - _NODES[a]) * 100.0
            for a, b in _BARS
        ])

        # Global stiffness assembly
        K = np.zeros((dof, dof))
        for k, (a, b) in enumerate(_BARS):
            xa, ya = _NODES[a] * 100.0
            xb, yb = _NODES[b] * 100.0
            L = L_bars[k]
            c = (xb - xa) / L
            s = (yb - ya) / L
            ke_local = (_E * A[k] / L) * np.array([
                [ c*c,  c*s, -c*c, -c*s],
                [ c*s,  s*s, -c*s, -s*s],
                [-c*c, -c*s,  c*c,  c*s],
                [-c*s, -s*s,  c*s,  s*s],
            ])
            dofs = [2*a, 2*a+1, 2*b, 2*b+1]
            for i, di in enumerate(dofs):
                for j, dj in enumerate(dofs):
                    K[di, dj] += ke_local[i, j]

        # External load vector: -P (down) at nodes 1 and 3
        F = np.zeros(dof)
        F[2*1 + 1] = -_P    # node 1, y-direction
        F[2*3 + 1] = -_P    # node 3, y-direction

        # Boundary conditions: fix nodes 4 (x,y) and 5 (x,y)
        fixed_dofs = [2*4, 2*4+1, 2*5, 2*5+1]
        free_dofs = [i for i in range(dof) if i not in fixed_dofs]

        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f  = F[free_dofs]

        try:
            u_free = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            return 1e9   # singular → infeasible

        u = np.zeros(dof)
        for i, di in enumerate(free_dofs):
            u[di] = u_free[i]

        # Bar stresses
        stresses = np.zeros(n_bars)
        for k, (a, b) in enumerate(_BARS):
            xa, ya = _NODES[a] * 100.0
            xb, yb = _NODES[b] * 100.0
            L = L_bars[k]
            c = (xb - xa) / L
            s = (yb - ya) / L
            delta_u = (u[2*b] - u[2*a]) * c + (u[2*b+1] - u[2*a+1]) * s
            stresses[k] = _E * delta_u / L

        # Nodal displacements (magnitude)
        disps = np.array([
            np.sqrt(u[2*i] ** 2 + u[2*i + 1] ** 2) for i in range(n_nodes)
        ])

        # Objective: total weight
        weight = float(_RHO * np.sum(A * L_bars))

        # Penalty for constraint violations
        sigma_max = 25.0   # kN/cm²
        disp_max  = 2.0    # cm
        pen_stress = np.sum(np.maximum(0.0, np.abs(stresses) - sigma_max) ** 2)
        pen_disp   = np.sum(np.maximum(0.0, disps - disp_max) ** 2)
        penalty = 1e4 * (pen_stress + pen_disp)

        return weight + penalty


# ============================================================================
# P3 — Container Placement Latency Minimisation (30-node cluster)
# ============================================================================

# 15 microservices to be assigned to 30 nodes
# x ∈ {0,…,29}^15  (integer; SFCM uses continuous relaxation + rounding)
# Constraints: CPU ≤80%, memory ≤90%, dependency hops ≤ 2

P3_BOUNDS = np.array([[0.0, 29.0]] * 15)   # continuous; rounded to int
P3_BUDGET = 1000

# Microservice properties (CPU demand, memory demand, dependency edges)
_MS_CPU  = np.array([0.12, 0.08, 0.20, 0.15, 0.10,
                     0.18, 0.07, 0.25, 0.11, 0.09,
                     0.14, 0.16, 0.22, 0.13, 0.19])   # fraction of node capacity
_MS_MEM  = np.array([0.15, 0.10, 0.25, 0.18, 0.12,
                     0.20, 0.09, 0.30, 0.14, 0.11,
                     0.16, 0.19, 0.28, 0.15, 0.22])   # fraction of node capacity
# Dependency pairs (microservice i → microservice j  with latency weight)
_DEPS = [
    (0,1,1.2), (0,2,0.8), (1,3,1.5), (2,3,0.9), (2,4,1.1),
    (3,5,2.0), (4,5,1.3), (5,6,0.7), (6,7,1.8), (7,8,1.0),
    (8,9,1.4), (9,10,0.6), (10,11,1.2), (11,12,1.7), (12,13,0.8),
    (13,14,1.1), (0,14,2.5), (6,10,1.3),
]
# Node network topology — fully connected grid with hop distance
# Nodes arranged in 5×6 grid; hop = max(|Δrow|, |Δcol|)
_NODE_ROWS = np.array([i // 6 for i in range(30)])
_NODE_COLS = np.array([i  % 6 for i in range(30)])


class ContainerPlacement:
    """
    P3: Container placement latency minimisation.

    evaluate(x) -> mean round-trip latency [ms]
    x is treated as continuous and rounded to integer node assignments.
    """

    bounds = P3_BOUNDS
    budget = P3_BUDGET
    d = 15

    # Node base latency (heterogeneous hardware)
    _BASE_LAT = np.array([
        0.5 + 0.1 * (i % 6) + 0.05 * np.sin(i) for i in range(30)
    ])

    def evaluate(self, x: np.ndarray) -> float:
        assignment = np.clip(np.round(x).astype(int), 0, 29)
        return self._simulate(assignment)

    # ------------------------------------------------------------------
    def _simulate(self, assignment: np.ndarray) -> float:
        """
        Discrete-event latency simulator.
        Returns mean round-trip latency across all dependency edges (ms).
        Applies penalty for constraint violations.
        """
        n_ms = 15
        n_nodes = 30

        # Resource utilisation per node
        cpu_util  = np.zeros(n_nodes)
        mem_util  = np.zeros(n_nodes)
        for i in range(n_ms):
            nd = assignment[i]
            cpu_util[nd] += _MS_CPU[i]
            mem_util[nd] += _MS_MEM[i]

        # Latency from dependency graph
        total_lat = 0.0
        n_deps = len(_DEPS)
        for src, dst, weight in _DEPS:
            ns = assignment[src]
            nd = assignment[dst]
            # Hop distance on the grid topology
            hop = int(max(
                abs(_NODE_ROWS[ns] - _NODE_ROWS[nd]),
                abs(_NODE_COLS[ns] - _NODE_COLS[nd]),
            ))
            # Base + hop + node processing
            lat = (self._BASE_LAT[ns] + self._BASE_LAT[nd]
                   + hop * 0.8 + weight * 0.5)
            total_lat += lat

        mean_lat = total_lat / n_deps   # ms

        # Constraint penalties
        cpu_viol = np.sum(np.maximum(0.0, cpu_util - 0.80) ** 2)
        mem_viol = np.sum(np.maximum(0.0, mem_util - 0.90) ** 2)

        # Hop constraint: each dep edge must use ≤ 2 hops
        hop_viol = 0.0
        for src, dst, _ in _DEPS:
            ns, nd = assignment[src], assignment[dst]
            hop = max(abs(_NODE_ROWS[ns] - _NODE_ROWS[nd]),
                      abs(_NODE_COLS[ns] - _NODE_COLS[nd]))
            hop_viol += max(0, hop - 2) ** 2

        penalty = 50.0 * (cpu_viol + mem_viol + hop_viol)
        return float(mean_lat + penalty)


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------

def get_real_world_problems(simulate_p1: bool = True) -> dict:
    """
    Returns dict of problem instances keyed by paper name.
    """
    return {
        "P1_ResNet18_HPO": ResNet18HPO(simulate=simulate_p1),
        "P2_TrussDesign":  TrussDesign(),
        "P3_ContainerPlacement": ContainerPlacement(),
    }
