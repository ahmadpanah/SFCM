"""
sfcm/core.py  —  SFCM main optimisation loop  (Algorithm 1 & 2)
All hyperparameters match Appendix C Table C1 (authoritative source).
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Optional
from .surrogate import EnsembleMLP, SurrogatePopulation
from .operators import (
    latin_hypercube_init, de_rand_1_bin,
    misprediction_density, noise_aware_score, trust_score,
)

DEFAULT_HPARAMS = dict(
    N=30, M=10,
    alpha=2.0,        # trust sensitivity
    kappa=1.5,        # UCB coefficient        [corrected from §4.4 draft 0.5]
    beta=0.8,         # mispred amplification  [corrected from §4.4 draft 1.5]
    lam=0.7,          # surrogate meta-fitness mixing λ
    tau0=0.3,         # initial trust threshold[corrected from §4.4 draft 0.4]
    delta0=0.1,       # initial replacement tolerance
    eps=1e-8,
    n0_factor=5,      # initial archive: n0 = n0_factor * d
    K_retrain=10,     # retrain every K generations
    K_new=5,          # or after K_new new true evals
    val_split=0.20,
    # ablation flags (default all False = full SFCM)
    no_dual_pop=False,
    no_noise_aware=False,
    no_mispred_explore=False,
    surrogate_off=False,
)


class SFCM:
    """
    Surrogate-Feedback Co-evolutionary Metaheuristic.

    Parameters
    ----------
    objective : callable  f(x) -> float   (minimisation)
    bounds    : (d, 2) array  [[lb, ub], ...]
    budget    : int  maximum true evaluations B
    seed      : int
    **kwargs  : override any key in DEFAULT_HPARAMS
    """

    def __init__(self, objective: Callable, bounds, budget: int,
                 seed: int = 0, **kwargs):
        self.f      = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.d      = self.bounds.shape[0]
        self.B      = budget
        self.rng    = np.random.default_rng(seed)
        self.hp     = {**DEFAULT_HPARAMS, **kwargs}

        self.archive_X: list = []
        self.archive_y: list = []
        self.n_true   = 0
        self.best_x: Optional[np.ndarray] = None
        self.best_y   = np.inf
        self.history_best:  list = []
        self.history_n_true:list = []

    # ── public ──────────────────────────────────────────────────────────
    def run(self):
        hp = self.hp
        N, M  = hp["N"], hp["M"]
        n0    = hp["n0_factor"] * self.d

        # 1. Initialisation
        P_X = latin_hypercube_init(N, self.bounds, self.rng)
        self._bulk_eval(P_X[:min(n0, self.B)])

        if not hp["surrogate_off"] and not hp["no_dual_pop"]:
            surr_pop = SurrogatePopulation(M, self.d, self.rng)
            surr_pop.fit_all(self.archive_X, self.archive_y)
            best_surr = surr_pop.best()
        elif not hp["surrogate_off"]:
            # noDP: single static surrogate
            best_surr = self._make_single_surrogate()
            best_surr.fit(self.archive_X, self.archive_y)
            surr_pop = None
        else:
            best_surr = None; surr_pop = None

        tau_t   = hp["tau0"]
        delta_t = hp["delta0"]
        gen_since = 0
        n_at_retrain = self.n_true

        while self.n_true < self.B:
            # -- resource allocation & effective fitness --
            f_eff     = np.zeros(N)
            sigma_hat = np.ones(N)
            pi        = np.zeros(N, dtype=bool)

            eps_bar = (best_surr.mean_abs_error(self.archive_X, self.archive_y)
                       if best_surr else 1.0)

            for i, x in enumerate(P_X):
                if best_surr and not hp["surrogate_off"]:
                    y_hat, sig = best_surr.predict(x)
                    sigma_hat[i] = sig
                    tr = trust_score(sig, eps_bar, hp["alpha"], hp["eps"])
                    if tr < tau_t:
                        pi[i] = True
                        f_eff[i] = self._true_eval(x)
                    else:
                        f_eff[i] = y_hat
                else:
                    f_eff[i] = self._true_eval(x); pi[i] = True

            # -- misprediction density --
            if best_surr and not hp["no_mispred_explore"]:
                rho = misprediction_density(
                    P_X, self.archive_X, self.archive_y,
                    best_surr, self._bw()
                )
                omega = rho / (rho.max() + hp["eps"])
            else:
                omega = np.zeros(N)

            # -- offspring generation --
            P_X_new = np.empty_like(P_X)
            for i in range(N):
                delta = de_rand_1_bin(P_X, i, self.bounds, self.rng)
                step  = (1.0 + hp["beta"] * omega[i]) * delta
                x_new = np.clip(P_X[i] + step, self.bounds[:,0], self.bounds[:,1])

                if best_surr and not hp["surrogate_off"]:
                    y_hat_n, sig_n = best_surr.predict(x_new)
                    tr_n = trust_score(sig_n, eps_bar, hp["alpha"], hp["eps"])
                    if tr_n < tau_t and self.n_true < self.B:
                        f_new = self._true_eval(x_new)
                    else:
                        f_new = y_hat_n
                else:
                    if self.n_true < self.B:
                        f_new = self._true_eval(x_new)
                    else:
                        f_new = f_eff[i]
                    sig_n = 0.0

                # noise-aware replacement
                if hp["no_noise_aware"]:
                    replace = f_new <= f_eff[i]
                else:
                    u_new = sig_n  if not pi[i] else 0.0
                    u_old = sigma_hat[i] if not pi[i] else 0.0
                    sn = noise_aware_score(f_new,   u_new, hp["kappa"])
                    so = noise_aware_score(f_eff[i],u_old, hp["kappa"])
                    replace = sn + delta_t < so

                P_X_new[i] = x_new if replace else P_X[i]
                if replace:
                    f_eff[i] = f_new; sigma_hat[i] = sig_n

            P_X = P_X_new

            # -- surrogate co-evolution --
            gen_since += 1
            n_new = self.n_true - n_at_retrain
            do_retrain = (gen_since >= hp["K_retrain"] or n_new >= hp["K_new"])

            if surr_pop is not None and do_retrain:
                surr_pop.evolve(self.archive_X, self.archive_y, P_X, hp["lam"])
                best_surr = surr_pop.best()
                gen_since = 0; n_at_retrain = self.n_true
            elif best_surr is not None and surr_pop is None and do_retrain:
                best_surr.fit(self.archive_X, self.archive_y)
                gen_since = 0; n_at_retrain = self.n_true

            # -- adaptive threshold decay --
            tau_t   = hp["tau0"]   * (self.B - self.n_true) / self.B
            delta_t = hp["delta0"] * (1.0 - self.n_true / self.B)

            self.history_best.append(self.best_y)
            self.history_n_true.append(self.n_true)

        return self.best_x, self.best_y

    # ── private ─────────────────────────────────────────────────────────
    def _true_eval(self, x):
        y = float(self.f(x))
        self.archive_X.append(x.copy())
        self.archive_y.append(y)
        self.n_true += 1
        if y < self.best_y:
            self.best_y = y; self.best_x = x.copy()
        return y

    def _bulk_eval(self, X):
        for x in X:
            if self.n_true < self.B:
                self._true_eval(x)

    def _bw(self):
        """Adaptive kernel bandwidth — Appendix B Eq.(B3)."""
        if len(self.archive_X) < 2: return 1.0
        X = np.stack(self.archive_X)
        diffs = X[:,None,:] - X[None,:,:]
        dists = np.linalg.norm(diffs, axis=-1)
        upper = dists[np.triu_indices(len(X), k=1)]
        return max(0.1 * float(np.median(upper)), 1e-6)

    def _make_single_surrogate(self):
        """For noDP ablation: single EnsembleMLP with default params."""
        return EnsembleMLP(d_in=self.d, seed=0)
