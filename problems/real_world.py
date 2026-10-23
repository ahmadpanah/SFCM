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
P1 requires PyTorch + torchvision (CUDA recommended).  To avoid a GPU
dependency during unit tests / CI, a light-weight POLYNOMIAL PROXY
(simulate=True) is provided.  **The polynomial proxy is a synthetic test
stand-in — it must never be used to produce, check or reproduce the P1 results
claimed in the manuscript (§4.2.2 / Table 3), which were obtained from
real ResNet-18 training (`simulate=False`, CIFAR-10, GPU).**  Any result
computed with simulate=True must be reported as "P1-polynomial-proxy".

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

# Label that any simulate=True result must carry — it is NOT the manuscript's
# ResNet-18/CIFAR-10 experiment (see ResNet18HPO docstring).
P1_PROXY_LABEL = "P1-polynomial-proxy"


class ResNet18HPO:
    """
    P1: ResNet-18 HPO on CIFAR-10.

    Parameters
    ----------
    simulate : bool
        If True, evaluates a fast POLYNOMIAL PROXY (synthetic test
        stand-in for CI / unit tests only — NOT a reproduction of the
        manuscript's P1 experiments, and never citable as such; label any
        such results "P1-polynomial-proxy").
        Set simulate=False for the actual experiment of the manuscript,
        which trains a real ResNet-18 on CIFAR-10 (requires CUDA); that real
        evaluator (``_train_real``) is the one defined in the manuscript §4.2.2.
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
        ⚠️  POLYNOMIAL PROXY — synthetic test simulation, NOT real training.

        This is a hand-crafted quadratic response surface used only for unit
        tests / CI.  It mimics the rough shape of an accuracy landscape,
        best proxy accuracy ≈ 93.5% at known-good hyperparameters, and its
        noise term is drawn from the ambient RNG (unseeded).  Results from
        this method are NOT the manuscript's ResNet-18/CIFAR-10 experiments
        and must be labelled "P1-polynomial-proxy" if they are ever reported.
        Returns 1 - proxy accuracy (minimisation).
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

# Standard benchmark: Schmit (1960), widely reproduced in the literature
# (canonical 10-bar planar truss, cf. Lingyun et al. 2005).
# d = 10 cross-sectional areas A_i — the paper (§4.2.2) quotes the range as
#     [0.1, 35.0] in²  =  [0.645, 225.8] cm²;  the decision-variable numeric
#     range is kept [0.1, 35.0] and is interpreted in in² (canonical benchmark).
# Objective  : minimise total weight  W = ρ Σ Aᵢ Lᵢ          [kg]
# Constraints: |σᵢ| ≤ 25 ksi = 172.37 MPa   (member stress)
#              |δⱼ| ≤ 2.0 in  = 0.0508 m    (nodal displacement)
#
# UNITS — REVISED FOR FIDELITY: the manuscript reports W in kilograms and
# quotes E = 10⁷ psi (68.95 GPa), ρ = 0.1 lb/in³ (2,767.99 kg/m³),
# downward loads of 100 kips (444.82 kN) and limits of 25 ksi / 2.0 in.
# The original implementation mixed kN/cm², kN/cm³ and a 100-kN load, so its
# objective could not produce the paper's kg-scale results (2312 kg etc.).
# This evaluator now works internally in SI (N, m, kg):
#     E    = 68.95 GPa         (10⁷ psi)
#     ρ    = 2,767.99 kg/m³   (0.1 lb/in³)
#     P    = 444.82 kN         (100 kips)
#     σ_max= 172.37 MPa        (25 ksi)
#     δ_max= 0.0508 m          (2.0 in)

P2_BOUNDS = np.array([[0.1, 35.0]] * 10)   # A_i in in² (canonical benchmark)
P2_BUDGET = 500

# Geometry (m): 10 nodes, 10 bars, horizontal span L=9.144m, height H=9.144m
_L = 9.144         # m
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
_RHO_SI = 2767.99          # kg/m³   (0.1 lb/in³, manuscript §4.2.2)
_E_SI   = 68.95e9          # Pa      (68.95 GPa = 10⁷ psi)
_P_SI   = 444.82e3         # N       (100 kips, downward at lower nodes 1 & 3)
_SIGMA_MAX_SI = 172.37e6   # Pa      (25 ksi)
_DISP_MAX_SI  = 0.0508     # m       (2.0 in)
# in² → m² conversion for the decision variables
IN2_TO_M2 = 6.4516e-4


class TrussDesign:
    """
    P2: 10-bar planar truss weight minimisation.

    evaluate(x) -> structural weight [kg]   subject to stress and
    displacement constraints; constraint violations are handled via an
    exterior penalty; the manuscript reports final designs as 100% feasible
    (ϕ(x⋆) = 0).
    """

    bounds = P2_BOUNDS
    budget = P2_BUDGET
    d = 10

    def evaluate(self, x: np.ndarray) -> float:
        # x: A_i in in² (decision variables, canonical benchmark range)
        #    → converted to m² for the SI FEM
        A = np.clip(x, 0.1, 35.0) * IN2_TO_M2              # in² → m²
        return self._fem_objective(A)

    # ------------------------------------------------------------------
    @staticmethod
    def _fem_objective(A: np.ndarray) -> float:
        """
        Simple 2D truss FEM in SI units (A in m², lengths in m,
        E in Pa, load in N).  Returns weight [kg] + penalty for violations.
        """
        n_bars = 10
        n_nodes = 6
        dof = 2 * n_nodes   # 12 DOFs

        # Bar lengths [m]
        L_bars = np.array([
            np.linalg.norm(_NODES[b] - _NODES[a])
            for a, b in _BARS
        ])

        # Global stiffness assembly
        K = np.zeros((dof, dof))
        for k, (a, b) in enumerate(_BARS):
            xa, ya = _NODES[a]
            xb, yb = _NODES[b]
            L = L_bars[k]
            c = (xb - xa) / L
            s = (yb - ya) / L
            ke_local = (_E_SI * A[k] / L) * np.array([
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
        F[2*1 + 1] = -_P_SI    # node 1, y-direction
        F[2*3 + 1] = -_P_SI    # node 3, y-direction

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

        # Bar stresses [Pa]
        stresses = np.zeros(n_bars)
        for k, (a, b) in enumerate(_BARS):
            xa, ya = _NODES[a]
            xb, yb = _NODES[b]
            L = L_bars[k]
            c = (xb - xa) / L
            s = (yb - ya) / L
            delta_u = (u[2*b] - u[2*a]) * c + (u[2*b+1] - u[2*a+1]) * s
            stresses[k] = _E_SI * delta_u / L

        # Nodal displacements (magnitude) [m]
        disps = np.array([
            np.sqrt(u[2*i] ** 2 + u[2*i + 1] ** 2) for i in range(n_nodes)
        ])

        # Objective: total structural weight [kg]  (ρ · Σ Aᵢ·Lᵢ, SI)
        weight = float(_RHO_SI * np.sum(A * L_bars))

        # Penalty for constraint violations — normalised by the allowed limits
        # so that penalty magnitudes are dimensionless O(1) when violated.
        pen_stress = np.sum(np.maximum(
            0.0, np.abs(stresses) - _SIGMA_MAX_SI) ** 2) / _SIGMA_MAX_SI ** 2
        pen_disp = np.sum(np.maximum(
            0.0, disps - _DISP_MAX_SI) ** 2) / _DISP_MAX_SI ** 2
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
