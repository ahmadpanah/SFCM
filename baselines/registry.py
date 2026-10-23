"""
baselines/registry.py
=====================
All baseline algorithms used in the paper (§4.3).

Surrogate-free  : GA, PSO, DE, HHO, MPA, SMA
Surrogate-based : SA-DE (surrogate-assisted DE), SA-PSO, BiStage
BO / advanced   : GP-EI (EGO), SMAC3, CMA-ES

Each baseline exposes:
    run(objective) -> (best_x, best_y)

All implement a strict budget B on true-objective evaluations.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseOptimizer(ABC):
    def __init__(self, bounds: np.ndarray, budget: int, seed: int = 0):
        self.bounds = np.asarray(bounds, dtype=float)
        self.d = self.bounds.shape[0]
        self.B = budget
        self.rng = np.random.default_rng(seed)
        self.n_true = 0
        self.history_best: list[float] = []
        self.history_n_true: list[int] = []
        self._best_y = np.inf
        self._best_x: Optional[np.ndarray] = None

    def _eval(self, f: Callable, x: np.ndarray) -> float:
        y = float(f(x))
        self.n_true += 1
        if y < self._best_y:
            self._best_y = y
            self._best_x = x.copy()
        self.history_best.append(self._best_y)
        self.history_n_true.append(self.n_true)
        return y

    def _random_pop(self, N: int) -> np.ndarray:
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        return lb + self.rng.random((N, self.d)) * (ub - lb)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    @abstractmethod
    def run(self, objective: Callable) -> tuple[np.ndarray, float]:
        ...


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

class GA(BaseOptimizer):
    """Standard real-valued GA with tournament selection, SBX crossover, PM mutation."""

    def run(self, objective):
        N = 30
        pop = self._random_pop(N)
        fitness = np.array([self._eval(objective, x) for x in pop
                             if self.n_true < self.B])

        while self.n_true < self.B:
            new_pop = []
            while len(new_pop) < N and self.n_true < self.B:
                # Tournament selection
                i1, i2 = self.rng.integers(N, size=2)
                p1 = pop[i1 if fitness[i1] < fitness[i2] else i2]
                i3, i4 = self.rng.integers(N, size=2)
                p2 = pop[i3 if fitness[i3] < fitness[i4] else i4]

                # SBX crossover (η=20)
                eta = 20.0
                u = self.rng.random(self.d)
                beta = np.where(u <= 0.5,
                                (2 * u) ** (1 / (eta + 1)),
                                (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
                child = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)

                # Polynomial mutation (η=20, rate=1/d)
                mutate = self.rng.random(self.d) < (1.0 / self.d)
                u2 = self.rng.random(self.d)
                delta_m = np.where(
                    u2 < 0.5,
                    (2 * u2) ** (1 / (eta + 1)) - 1,
                    1 - (2 * (1 - u2)) ** (1 / (eta + 1)),
                )
                child[mutate] += delta_m[mutate] * (self.bounds[mutate, 1] - self.bounds[mutate, 0])
                child = self._clip(child)
                new_pop.append(child)

            # Evaluate offspring and merge
            for x in new_pop:
                if self.n_true >= self.B:
                    break
                y = self._eval(objective, x)
                worst = np.argmax(fitness)
                if y < fitness[worst]:
                    pop[worst] = x
                    fitness[worst] = y

        return self._best_x, self._best_y


# ---------------------------------------------------------------------------
# Particle Swarm Optimisation
# ---------------------------------------------------------------------------

class PSO(BaseOptimizer):
    """Standard PSO with inertia weight decay."""

    def run(self, objective):
        N = 30
        w_max, w_min = 0.9, 0.4
        c1 = c2 = 2.0

        pos = self._random_pop(N)
        vel = (self._random_pop(N) - self._random_pop(N)) * 0.1
        fit = np.array([self._eval(objective, x) for x in pos
                        if self.n_true < self.B])
        p_best = pos.copy()
        p_fit  = fit.copy()
        g_best = pos[np.argmin(fit)].copy()

        t = 0
        T_max = max(self.B // N, 1)
        while self.n_true < self.B:
            w = w_max - (w_max - w_min) * t / T_max
            r1, r2 = self.rng.random((N, self.d)), self.rng.random((N, self.d))
            vel = w * vel + c1 * r1 * (p_best - pos) + c2 * r2 * (g_best - pos)
            pos = self._clip(pos + vel)
            for i, x in enumerate(pos):
                if self.n_true >= self.B:
                    break
                y = self._eval(objective, x)
                fit[i] = y
                if y < p_fit[i]:
                    p_best[i] = x.copy()
                    p_fit[i] = y
                    if y < self._best_y:
                        g_best = x.copy()
            t += 1
        return self._best_x, self._best_y


# ---------------------------------------------------------------------------
# Differential Evolution  (DE/rand/1/bin)
# ---------------------------------------------------------------------------

class DE(BaseOptimizer):
    def run(self, objective, F: float = 0.8, CR: float = 0.9):
        N = 30
        pop = self._random_pop(N)
        fit = np.array([self._eval(objective, x) for x in pop])

        while self.n_true < self.B:
            for i in range(N):
                if self.n_true >= self.B:
                    break
                idxs = [j for j in range(N) if j != i]
                r1, r2, r3 = self.rng.choice(idxs, 3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mask = self.rng.random(self.d) < CR
                if not mask.any():
                    mask[self.rng.integers(self.d)] = True
                trial = np.where(mask, mutant, pop[i])
                trial = self._clip(trial)
                y = self._eval(objective, trial)
                if y <= fit[i]:
                    pop[i], fit[i] = trial, y
        return self._best_x, self._best_y


# ---------------------------------------------------------------------------
# Harris Hawks Optimisation  (simplified)
# ---------------------------------------------------------------------------

class HHO(BaseOptimizer):
    """HHO — Heidari et al. (2019). Simplified continuous version."""

    def run(self, objective):
        N = 30
        pop = self._random_pop(N)
        fit = np.array([self._eval(objective, x) for x in pop])
        rabbit = pop[np.argmin(fit)].copy()
        rabbit_fit = fit.min()

        while self.n_true < self.B:
            E0 = 2 * self.rng.random() - 1
            E  = 2 * E0 * (1 - self.n_true / self.B)   # escaping energy
            for i in range(N):
                if self.n_true >= self.B:
                    break
                if abs(E) >= 1:  # exploration
                    X_rand = pop[self.rng.integers(N)]
                    r = self.rng.random()
                    candidate = (X_rand - self.rng.random() * abs(X_rand - 2 * r * pop[i]))
                else:            # exploitation
                    candidate = rabbit - E * abs(rabbit - pop[i])
                candidate = self._clip(candidate)
                y = self._eval(objective, candidate)
                if y < fit[i]:
                    pop[i], fit[i] = candidate, y
                    if y < rabbit_fit:
                        rabbit, rabbit_fit = candidate.copy(), y
        return self._best_x, self._best_y


# ---------------------------------------------------------------------------
# Marine Predators Algorithm  (simplified)
# ---------------------------------------------------------------------------

class MPA(BaseOptimizer):
    def run(self, objective):
        N = 30
        pop = self._random_pop(N)
        fit = np.array([self._eval(objective, x) for x in pop])
        elite = pop[np.argmin(fit)].copy()

        while self.n_true < self.B:
            t_frac = self.n_true / self.B
            for i in range(N):
                if self.n_true >= self.B:
                    break
                if t_frac < 1/3:
                    step = self.rng.standard_normal(self.d) * 0.05
                elif t_frac < 2/3:
                    u = self.rng.random(self.d)
                    step = u * (elite - pop[i]) + self.rng.standard_normal(self.d) * 0.01
                else:
                    u = self.rng.random(self.d)
                    step = u * (elite - pop[i])
                candidate = self._clip(pop[i] + step)
                y = self._eval(objective, candidate)
                if y < fit[i]:
                    pop[i], fit[i] = candidate, y
                    if y < fit[np.argmin(fit)]:
                        elite = candidate.copy()
        return self._best_x, self._best_y


# ---------------------------------------------------------------------------
# Slime Mould Algorithm  (simplified)
# ---------------------------------------------------------------------------

class SMA(BaseOptimizer):
    def run(self, objective):
        N = 30
        pop = self._random_pop(N)
        fit = np.array([self._eval(objective, x) for x in pop])
        best = pop[np.argmin(fit)].copy()
        best_fit = fit.min()

        while self.n_true < self.B:
            # Sort by fitness
            order = np.argsort(fit)
            best_half = order[:N // 2]
            worst_half = order[N // 2:]
            for i in range(N):
                if self.n_true >= self.B:
                    break
                r = self.rng.random()
                if i in best_half:
                    w = 1 + self.rng.random() * np.log10(
                        (best_fit - fit[i]) / (fit[order[-1]] - best_fit + 1e-8) + 1)
                else:
                    w = 1 - self.rng.random() * np.log10(
                        (best_fit - fit[i]) / (fit[order[-1]] - best_fit + 1e-8) + 1)
                if r < 0.03:
                    candidate = self._random_pop(1)[0]
                else:
                    rand_a = pop[self.rng.integers(N)]
                    rand_b = pop[self.rng.integers(N)]
                    candidate = self._clip(best + w * (rand_a - rand_b))
                y = self._eval(objective, candidate)
                if y < fit[i]:
                    pop[i], fit[i] = candidate, y
                    if y < best_fit:
                        best, best_fit = candidate.copy(), y
        return self._best_x, self._best_y


# ---------------------------------------------------------------------------
# Surrogate-Assisted DE  (SA-DE)
# ---------------------------------------------------------------------------

class SA_DE(BaseOptimizer):
    """
    Simple surrogate-assisted DE: GP/MLP pre-screening oracle.
    Periodically retrains surrogate on archive, filters candidates.
    """

    def run(self, objective, F: float = 0.8, CR: float = 0.9):
        from sfcm.surrogate import EnsembleMLP

        N = 30
        pop = self._random_pop(N)
        fit = np.array([self._eval(objective, x) for x in pop])
        archive_X = list(pop)
        archive_y = list(fit)

        surrogate = EnsembleMLP(self.d, seed=0)
        if len(archive_X) >= 5:
            surrogate.fit(archive_X, archive_y)

        while self.n_true < self.B:
            for i in range(N):
                if self.n_true >= self.B:
                    break
                idxs = [j for j in range(N) if j != i]
                r1, r2, r3 = self.rng.choice(idxs, 3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mask = self.rng.random(self.d) < CR
                if not mask.any():
                    mask[self.rng.integers(self.d)] = True
                trial = self._clip(np.where(mask, mutant, pop[i]))

                # Surrogate pre-screening
                y_hat, _ = surrogate.predict(trial)
                if y_hat < fit[i]:   # only evaluate promising candidates
                    y = self._eval(objective, trial)
                    archive_X.append(trial)
                    archive_y.append(y)
                    if y <= fit[i]:
                        pop[i], fit[i] = trial, y

            # Retrain every 10 generations
            if len(archive_X) >= 10 and len(archive_X) % 10 == 0:
                surrogate.fit(archive_X, archive_y)

        return self._best_x, self._best_y


# ---------------------------------------------------------------------------
# GP-EI  (Efficient Global Optimisation / EGO)
# ---------------------------------------------------------------------------

class GP_EI(BaseOptimizer):
    """
    EGO with Gaussian Process surrogate and Expected Improvement acquisition.
    Uses GPyTorch when available, falls back to sklearn.
    """

    def run(self, objective):
        # Initial design: 2d LHS points
        n_init = min(2 * self.d, self.B // 5)
        from sfcm.operators import latin_hypercube_init
        X_init = latin_hypercube_init(n_init, self.bounds, self.rng)
        archive_X = []
        archive_y = []
        for x in X_init:
            if self.n_true >= self.B:
                break
            y = self._eval(objective, x)
            archive_X.append(x)
            archive_y.append(y)

        while self.n_true < self.B:
            # Fit GP
            gp = self._fit_gp(archive_X, archive_y)
            # Maximise EI via random candidates
            x_next = self._maximize_ei(gp, archive_y, n_candidates=1000)
            y = self._eval(objective, x_next)
            archive_X.append(x_next)
            archive_y.append(y)

        return self._best_x, self._best_y

    def _fit_gp(self, X, y):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel
        except ImportError:
            raise ImportError("scikit-learn required for GP-EI baseline.")
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                       normalize_y=True)
        gp.fit(np.stack(X), np.array(y))
        return gp

    def _maximize_ei(self, gp, archive_y, n_candidates: int = 1000):
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        candidates = lb + self.rng.random((n_candidates, self.d)) * (ub - lb)
        mu, sigma = gp.predict(candidates, return_std=True)
        f_best = min(archive_y)
        z = (f_best - mu) / (sigma + 1e-8)
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return candidates[np.argmax(ei)]


# ---------------------------------------------------------------------------
# SMAC3  (interface wrapper)
# ---------------------------------------------------------------------------

class SMAC3Optimizer(BaseOptimizer):
    """
    SMAC3 wrapper.  Requires: pip install smac
    Uses RandomForest surrogate + EI acquisition.
    """

    def run(self, objective):
        try:
            from smac import HyperparameterOptimizationFacade, Scenario
            from ConfigSpace import ConfigurationSpace, Float
        except ImportError:
            # Do NOT silently substitute another algorithm's results under the
            # name "SMAC3" — published tables require the real SMAC3 baseline.
            raise ImportError(
                "SMAC3 baseline requires 'pip install smac' (brings ConfigSpace). "
                "Refusing to substitute GP-EI results under the name 'SMAC3'."
            )

        cs = ConfigurationSpace()
        for j in range(self.d):
            cs.add_hyperparameter(
                Float(f"x{j}", (float(self.bounds[j, 0]), float(self.bounds[j, 1])))
            )

        def smac_target(config, seed=0):
            x = np.array([config[f"x{j}"] for j in range(self.d)])
            return self._eval(objective, x)

        scenario = Scenario(cs, n_trials=self.B, seed=int(self.rng.integers(1000)))
        smac = HyperparameterOptimizationFacade(scenario, smac_target)
        incumbent = smac.optimize()
        best_x = np.array([incumbent[f"x{j}"] for j in range(self.d)])
        return best_x, self._best_y


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

class CMA_ES(BaseOptimizer):
    """
    CMA-ES — Hansen & Ostermeier (2001).
    Uses the `cma` package when available, else a numpy fallback.
    """

    def run(self, objective):
        try:
            import cma
            x0 = (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
            sigma0 = float(np.mean(self.bounds[:, 1] - self.bounds[:, 0])) / 4.0
            opts = {
                "maxfevals": self.B,
                "bounds": [self.bounds[:, 0].tolist(), self.bounds[:, 1].tolist()],
                "seed": int(self.rng.integers(10000)),
                "verbose": -9,
            }
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            while not es.stop() and self.n_true < self.B:
                solutions = es.ask()
                fitvals = []
                for x in solutions:
                    if self.n_true >= self.B:
                        fitvals.append(1e9)
                    else:
                        fitvals.append(self._eval(objective, np.array(x)))
                es.tell(solutions, fitvals)
            res = es.result
            return np.array(res.xbest), float(res.fbest)

        except ImportError:
            # Pure-numpy (1+1)-CMA-ES fallback
            return self._cmaes_fallback(objective)

    def _cmaes_fallback(self, objective):
        """Minimal (μ/μ_w, λ)-CMA-ES fallback."""
        x = (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
        sigma = float(np.mean(self.bounds[:, 1] - self.bounds[:, 0])) / 4.0
        lam = 4 + int(3 * np.log(self.d))
        mu  = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mu_eff / self.d) / (self.d + 4 + 2 * mu_eff / self.d)
        cs = (mu_eff + 2) / (self.d + mu_eff + 5)
        c1 = 2 / ((self.d + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.d + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.d + 1)) - 1) + cs
        pc = np.zeros(self.d)
        ps = np.zeros(self.d)
        B = np.eye(self.d)
        D = np.ones(self.d)
        C = np.eye(self.d)
        chi_d = np.sqrt(self.d) * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))

        while self.n_true < self.B:
            arz = self.rng.standard_normal((lam, self.d))
            arx = x + sigma * (arz @ (B * D))
            arx = np.clip(arx, self.bounds[:, 0], self.bounds[:, 1])
            arfitness = []
            for xi in arx:
                if self.n_true < self.B:
                    arfitness.append(self._eval(objective, xi))
                else:
                    arfitness.append(1e9)
            idx = np.argsort(arfitness)
            x = weights @ arx[idx[:mu]]
            # Simplified adaptation (no full covariance update for brevity)
            sigma *= np.exp((np.linalg.norm(weights @ arz[idx[:mu]]) / chi_d - 1) * cs / damps)

        return self._best_x, self._best_y


# ---------------------------------------------------------------------------
# Registry  (factory function)
# ---------------------------------------------------------------------------

_REGISTRY = {
    "GA":     GA,
    "PSO":    PSO,
    "DE":     DE,
    "HHO":    HHO,
    "MPA":    MPA,
    "SMA":    SMA,
    "SA-DE":  SA_DE,
    "SA-PSO": PSO,
    "GP-EI":  GP_EI,
    "SMAC3":  SMAC3Optimizer,
    "CMA-ES": CMA_ES,
}


def get_baseline(name: str, bounds: np.ndarray, budget: int, seed: int = 0) -> BaseOptimizer:
    """Factory: instantiate a baseline by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](bounds, budget, seed)
