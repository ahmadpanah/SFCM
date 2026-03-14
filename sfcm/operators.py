"""
sfcm/operators.py
=================
All primitive operators used by SFCM.

Each function maps directly to a numbered equation in the paper:

  latin_hypercube_init  — §4.4 (LHS initialisation)
  trust_score           — Eq.(4)
  effective_fitness     — Eq.(5)
  noise_aware_score     — Eq.(6)
  robust_replace        — Eq.(7)
  misprediction_density — Eq.(8)
  scaled_perturbation   — Eq.(9)
  de_rand_1_bin         — §3.2 / Appendix A (base MH operator)
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .surrogate import SurrogateConfig


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def latin_hypercube_init(
    N: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Latin hypercube sampling over the bounded domain Ω.
    Produces N points in [lb, ub]^d with maximin stratification.
    Ref: §4.4, McKay et al. (1979).
    """
    d = bounds.shape[0]
    # One sample per stratum per dimension
    intervals = np.linspace(0.0, 1.0, N + 1)
    lower, upper = intervals[:-1], intervals[1:]
    points = rng.uniform(lower, upper, size=(d, N)).T  # (N, d)

    # Permute each dimension independently
    for j in range(d):
        perm = rng.permutation(N)
        points[:, j] = points[perm, j]

    # Scale to bounds
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    return lb + points * (ub - lb)


# ---------------------------------------------------------------------------
# Trust score  —  Eq.(4)
# ---------------------------------------------------------------------------

def trust_score(
    sigma_hat: float,
    eps_bar: float,
    alpha: float,
    eps: float = 1e-8,
) -> float:
    """
    trust_t(x) = exp( -α · σ̂(x) / (ε̄_t + ε) )    Eq.(4)

    Parameters
    ----------
    sigma_hat : predictive uncertainty at x
    eps_bar   : mean absolute prediction error ε̄_t on archive
    alpha     : sensitivity parameter (default 2.0)
    eps       : numerical stability constant (default 1e-8)
    """
    return float(np.exp(-alpha * sigma_hat / (eps_bar + eps)))


# ---------------------------------------------------------------------------
# Effective fitness  —  Eq.(5)
# ---------------------------------------------------------------------------

def effective_fitness(
    pi: bool,
    f_true: float,
    y_hat: float,
) -> float:
    """
    f̃_t(x) = π_t(x)·f(x) + (1 − π_t(x))·ŷ(x)    Eq.(5)
    """
    return f_true if pi else y_hat


# ---------------------------------------------------------------------------
# UCB noise-aware score  —  Eq.(6)
# ---------------------------------------------------------------------------

def noise_aware_score(
    f_eff: float,
    u: float,
    kappa: float,
) -> float:
    """
    score_t(x) = f̃_t(x) − κ · u_t(x)    Eq.(6)

    u_t(x) = σ̂(x)  when surrogate is used (π=0), else 0.
    κ = 1.5  (Appendix C Table C1)
    """
    return f_eff - kappa * u


# ---------------------------------------------------------------------------
# Robust replacement rule  —  Eq.(7)
# ---------------------------------------------------------------------------

def robust_replace(
    score_new: float,
    score_old: float,
    delta_t: float,
) -> bool:
    """
    Replace iff  score_new + δ_t  <  score_old    Eq.(7)

    δ_t decays linearly from δ_0=0.1 → 0 as budget is consumed.
    """
    return score_new + delta_t < score_old


# ---------------------------------------------------------------------------
# Misprediction density  —  Eq.(8)
# ---------------------------------------------------------------------------

def misprediction_density(
    query_points: np.ndarray,
    archive_X: list[np.ndarray],
    archive_y: list[float],
    surrogate: "SurrogateConfig",
    h: float,
) -> np.ndarray:
    """
    ρ_t(x) = Σ_{(xᵢ,yᵢ)∈D_t}  eᵢ · K_h(x − xᵢ)    Eq.(8)

    Gaussian kernel  K_h(z) = (2πh²)^{-1/2} exp(−‖z‖²/2h²)
    eᵢ = |s(xᵢ; θ̂_t) − yᵢ|

    Parameters
    ----------
    query_points : (N, d) array — P^X_t
    archive_X    : list of archived inputs
    archive_y    : list of archived true values
    surrogate    : current best surrogate
    h            : kernel bandwidth (adaptive, Appendix B Eq.(B3))

    Returns
    -------
    rho : (N,) array of misprediction density at each query point
    """
    if len(archive_X) == 0:
        return np.zeros(len(query_points))

    Xa = np.stack(archive_X)                    # (n, d)
    errors = np.array([
        abs(surrogate.predict(xi)[0] - yi)
        for xi, yi in zip(archive_X, archive_y)
    ])                                           # (n,)

    rho = np.zeros(len(query_points))
    h2 = h ** 2
    for i, x in enumerate(query_points):
        diff = Xa - x[None, :]                  # (n, d)
        sq_dist = np.sum(diff ** 2, axis=1)     # (n,)
        kernel_vals = np.exp(-sq_dist / (2.0 * h2))
        rho[i] = float(np.dot(errors, kernel_vals))

    return rho


# ---------------------------------------------------------------------------
# Scaled perturbation (misprediction-guided)  —  Eq.(9)
# ---------------------------------------------------------------------------

def scaled_perturbation(
    base_delta: np.ndarray,
    omega: float,
    beta: float,
) -> np.ndarray:
    """
    x' = x + (1 + β · ω_t(x)) · Δ_t(x)    Eq.(9)

    Parameters
    ----------
    base_delta : Δ_t(x) from DE/rand/1/bin
    omega      : normalised misprediction weight ρ_t(x)/max ρ_t
    beta       : amplification strength (default 0.8, Appendix C)
    """
    return (1.0 + beta * omega) * base_delta


# ---------------------------------------------------------------------------
# DE/rand/1/bin base mutation  —  §3.2
# ---------------------------------------------------------------------------

def de_rand_1_bin(
    population: np.ndarray,
    target_idx: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
    F: float = 0.8,
    CR: float = 0.9,
) -> np.ndarray:
    """
    DE/rand/1/bin mutation and crossover.

    Δ_t(x_i) = F · (x_r2 − x_r3)   (mutation, r1 unused as rand base)

    Returns the *difference vector* Δ so that SFCM's Eq.(9) can scale it.
    The caller adds it to the parent: x' = x + scaled_perturbation(Δ, ω, β).

    Parameters
    ----------
    population  : (N, d) current solution population
    target_idx  : index of the target vector
    bounds      : (d, 2) lower/upper bounds
    rng         : numpy Generator
    F           : DE scale factor (default 0.8)
    CR          : crossover rate (default 0.9)
    """
    N, d = population.shape
    idxs = [i for i in range(N) if i != target_idx]
    r1, r2, r3 = rng.choice(idxs, 3, replace=False)

    mutant = population[r1] + F * (population[r2] - population[r3])

    # Crossover mask
    mask = rng.random(d) < CR
    if not mask.any():
        mask[rng.integers(d)] = True

    # Return only the *delta* so the caller can apply Eq.(9) scaling
    base_delta = np.where(mask, mutant - population[target_idx], 0.0)
    return base_delta
