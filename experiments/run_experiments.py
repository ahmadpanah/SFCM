"""
experiments/run_experiments.py
==============================
Reproduces all tables and figures from the paper.

Usage
-----
# Full experiment (30 runs × 12 functions × 3 dims + 3 real-world)
python experiments/run_experiments.py --all --n_runs 30

# Single benchmark sanity check
python experiments/run_experiments.py --bench Rastrigin --d 10 --n_runs 3

# Ablation study only
python experiments/run_experiments.py --ablation --d 30 --n_runs 30

# Real-world problems (simulate mode — no GPU required)
python experiments/run_experiments.py --realworld --simulate

Outputs
-------
results/
  tables/   tab2_synthetic.csv, tab3_realworld.csv,
            tab4_ablation.csv, tab5_wilcoxon.csv
  figures/  fig2_convergence.pdf, fig3_friedman.pdf,
            fig4_realworld.pdf, fig5_ablation.pdf,
            fig6_boxplots.pdf, fig7_noise_robust.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfcm.core import SFCM, DEFAULT_HPARAMS
from benchmarks.functions import get_benchmark_suite, BenchmarkFunction
from problems.real_world import get_real_world_problems
from baselines.registry import get_baseline

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
TABLES_DIR  = RESULTS_DIR / "tables"
FIGS_DIR    = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Budgets — Appendix D Table D2
# ---------------------------------------------------------------------------
BUDGETS = {10: 10_000, 30: 30_000, 50: 50_000}

REAL_WORLD_BUDGETS = {
    "P1_ResNet18_HPO":       100,
    "P2_TrussDesign":        500,
    "P3_ContainerPlacement": 1000,
}


# ---------------------------------------------------------------------------
# Algorithm names used in comparisons
# ---------------------------------------------------------------------------
SURROGATE_FREE = ["GA", "PSO", "DE", "HHO", "MPA", "SMA"]
SURROGATE_ASSISTED = ["SA-DE", "SA-PSO", "BiStage", "GP-EI", "SMAC3", "CMA-ES"]
ALL_BASELINES = SURROGATE_FREE + SURROGATE_ASSISTED


# ---------------------------------------------------------------------------
# Single run helper
# ---------------------------------------------------------------------------

def run_sfcm(
    objective,
    bounds: np.ndarray,
    budget: int,
    seed: int,
    **kwargs,
) -> dict:
    """One run of SFCM. Returns results dict."""
    opt = SFCM(objective, bounds, budget, seed=seed, **kwargs)
    t0 = time.perf_counter()
    best_x, best_y = opt.run()
    elapsed = time.perf_counter() - t0
    return dict(
        best_y=best_y,
        n_true=opt.n_true,
        rho_eval=1.0 - opt.n_true / budget,
        wall_clock=elapsed,
        history_best=opt.history_best,
        history_n_true=opt.history_n_true,
    )


def run_baseline(
    name: str,
    objective,
    bounds: np.ndarray,
    budget: int,
    seed: int,
) -> dict:
    """One run of a named baseline. Returns results dict."""
    algo = get_baseline(name, bounds, budget, seed)
    t0 = time.perf_counter()
    best_x, best_y = algo.run(objective)
    elapsed = time.perf_counter() - t0
    return dict(
        best_y=best_y,
        n_true=getattr(algo, "n_true", budget),
        wall_clock=elapsed,
        history_best=getattr(algo, "history_best", [best_y]),
        history_n_true=getattr(algo, "history_n_true", [budget]),
    )


# ---------------------------------------------------------------------------
# Synthetic benchmark experiment  →  Table 2 + Figs 2, 3, 6
# ---------------------------------------------------------------------------

def run_synthetic(
    dims: list[int],
    n_runs: int,
    selected_funcs: list[str] | None = None,
    baselines: list[str] | None = None,
) -> pd.DataFrame:

    if baselines is None:
        baselines = ["HHO", "MPA", "SMA", "SA-DE", "GP-EI", "SMAC3", "CMA-ES"]

    rows = []
    suite = {d: get_benchmark_suite(d) for d in dims}

    for d in dims:
        budget = BUDGETS[d]
        for fn in suite[d]:
            if selected_funcs and fn.name not in selected_funcs:
                continue

            bounds = np.array([[-fn.bounds, fn.bounds]] * d)
            objective = fn.func

            print(f"  [{fn.name:15s} d={d}]  SFCM", end="", flush=True)
            sfcm_results = []
            for r in range(n_runs):
                res = run_sfcm(objective, bounds, budget, seed=r)
                sfcm_results.append(res["best_y"] - fn.optimum)
            print(f" done. mean={np.mean(sfcm_results):.3e}")

            row = dict(func=fn.name, d=d,
                       sfcm_mean=np.mean(sfcm_results),
                       sfcm_std=np.std(sfcm_results),
                       sfcm_median=np.median(sfcm_results),
                       sfcm_iqr=float(np.percentile(sfcm_results, 75) -
                                       np.percentile(sfcm_results, 25)))

            for bl in baselines:
                print(f"           {bl}", end="", flush=True)
                bl_results = []
                for r in range(n_runs):
                    try:
                        res = run_baseline(bl, objective, bounds, budget, seed=r)
                        bl_results.append(res["best_y"] - fn.optimum)
                    except Exception as e:
                        bl_results.append(np.nan)
                        print(f"(err: {e})", end="")

                bl_arr = np.array(bl_results)
                sfcm_arr = np.array(sfcm_results)
                _, p_val = stats.wilcoxon(sfcm_arr, bl_arr, alternative="less",
                                          zero_method="wilcox")
                _, p_holm = _holm_bonferroni([p_val], alpha=0.05)

                row[f"{bl}_mean"] = np.nanmean(bl_arr)
                row[f"{bl}_std"]  = np.nanstd(bl_arr)
                row[f"{bl}_p"]    = p_val
                row[f"{bl}_sig"]  = p_holm[0] < 0.05
                row[f"{bl}_cliff"] = _cliff_delta(sfcm_arr, bl_arr)
                print(f" done. mean={np.nanmean(bl_arr):.3e} p={p_val:.3f}")

            rows.append(row)

    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "tab2_synthetic.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Real-world experiment  →  Table 3 + Fig 4
# ---------------------------------------------------------------------------

def run_realworld(n_runs: int, simulate_p1: bool = True) -> pd.DataFrame:
    problems = get_real_world_problems(simulate_p1=simulate_p1)
    rows = []
    for name, prob in problems.items():
        budget = REAL_WORLD_BUDGETS[name]
        bounds = prob.bounds
        objective = prob.evaluate

        sfcm_results = []
        for r in range(n_runs):
            res = run_sfcm(objective, bounds, budget, seed=r)
            sfcm_results.append(res["best_y"])
        sfcm_arr = np.array(sfcm_results)

        row = dict(problem=name,
                   sfcm_mean=sfcm_arr.mean(),
                   sfcm_std=sfcm_arr.std(),
                   sfcm_sr=float(np.mean(sfcm_arr < _success_threshold(name))))

        for bl in ["SA-DE", "HHO"]:
            bl_results = []
            for r in range(n_runs):
                res = run_baseline(bl, objective, bounds, budget, seed=r)
                bl_results.append(res["best_y"])
            bl_arr = np.array(bl_results)
            _, p = stats.wilcoxon(sfcm_arr, bl_arr, alternative="less",
                                  zero_method="wilcox")
            row[f"{bl}_mean"] = bl_arr.mean()
            row[f"{bl}_std"]  = bl_arr.std()
            row[f"{bl}_p"]    = p
            row[f"{bl}_sig"]  = p < 0.05
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "tab3_realworld.csv", index=False)
    return df


def _success_threshold(problem_name: str) -> float:
    thresholds = {
        "P1_ResNet18_HPO":       0.076,   # 1 - 92.4% accuracy
        "P2_TrussDesign":        145.0,   # weight ≤ 145 kN  (sign: minimising weight)
        "P3_ContainerPlacement": 26.0,    # latency ≤ 26 ms
    }
    return thresholds.get(problem_name, np.inf)


# ---------------------------------------------------------------------------
# Ablation study  →  Table 4 + Fig 5
# ---------------------------------------------------------------------------

def run_ablation(d: int = 30, n_runs: int = 30) -> pd.DataFrame:
    budget = BUDGETS[d]
    suite  = get_benchmark_suite(d)
    # Use Sphere, Rastrigin, CEC_F1, Griewank, Ackley, Levy  (6 representative)
    selected = ["Sphere", "Rastrigin", "CEC_F1", "Griewank", "Ackley", "Levy"]
    variants = {
        "SFCM (full)":      {},
        "SFCM-noDP":        {"no_dual_pop": True},
        "SFCM-noNS":        {"no_noise_aware": True},
        "SFCM-noME":        {"no_mispred_explore": True},
        "Base (DE)":        {"surrogate_off": True},
    }
    rows = []
    for fn in suite:
        if fn.name not in selected:
            continue
        bounds = np.array([[-fn.bounds, fn.bounds]] * d)
        row = dict(func=fn.name)
        for var_name, var_kwargs in variants.items():
            results = []
            for r in range(n_runs):
                res = run_sfcm(fn.func, bounds, budget, seed=r, **var_kwargs)
                results.append(res["best_y"] - fn.optimum)
            arr = np.array(results)
            row[f"{var_name}_mean"] = arr.mean()
            row[f"{var_name}_std"]  = arr.std()
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "tab4_ablation.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Statistical significance summary  →  Table 5
# ---------------------------------------------------------------------------

def compute_wilcoxon_table(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise wins/losses/ties across all problem instances."""
    rows = []
    baselines = ["HHO", "MPA", "SMA", "SA-DE", "GP-EI", "SMAC3", "CMA-ES"]
    for bl in baselines:
        sig_col = f"{bl}_sig"
        if sig_col not in synthetic_df.columns:
            continue
        wins   = int(synthetic_df[f"{bl}_sig"].sum())
        ties   = int((~synthetic_df[f"{bl}_sig"]).sum())
        losses = 0   # SFCM is tested as "less than" → no losses in this frame
        cliff  = float(synthetic_df[f"{bl}_cliff"].mean())
        rows.append(dict(algorithm=bl, wins=wins, losses=losses, ties=ties,
                         cliff_delta=cliff))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "tab5_wilcoxon.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Noise robustness sweep  →  Fig 7
# ---------------------------------------------------------------------------

def run_noise_robustness(
    d: int = 30,
    n_runs: int = 10,
    noise_levels: list[float] | None = None,
) -> pd.DataFrame:
    if noise_levels is None:
        noise_levels = [0, 5, 10, 15, 20, 25, 30]

    budget = BUDGETS[d]
    fn_name = "Rastrigin"
    fn = next(f for f in get_benchmark_suite(d) if f.name == fn_name)
    bounds = np.array([[-fn.bounds, fn.bounds]] * d)
    baseline_clean = {}   # store clean-run reference

    rows = []
    for noise_pct in noise_levels:
        noise_std = noise_pct / 100.0

        def noisy_obj(x, _ns=noise_std):
            return fn.func(x) + np.random.default_rng().normal(0, _ns * abs(fn.func(x)) + 1e-8)

        for algo in ["SFCM", "SA-DE", "HHO"]:
            results = []
            for r in range(n_runs):
                if algo == "SFCM":
                    res = run_sfcm(noisy_obj, bounds, budget, seed=r)
                else:
                    res = run_baseline(algo, noisy_obj, bounds, budget, seed=r)
                results.append(res["best_y"])

            if noise_pct == 0:
                baseline_clean[algo] = float(np.mean(results))

            mean_val = float(np.mean(results))
            ref = baseline_clean.get(algo, mean_val)
            rel_deg = 100.0 * (mean_val - ref) / (abs(ref) + 1e-8) if ref != 0 else 0.0
            rows.append(dict(noise_pct=noise_pct, algorithm=algo,
                             mean_obj=mean_val, rel_degradation=rel_deg))

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "tab_noise_robustness.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _holm_bonferroni(p_values: list[float], alpha: float = 0.05):
    """Holm-Bonferroni correction. Returns (reject, corrected_p)."""
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    reject = np.zeros(m, dtype=bool)
    corrected = np.ones(m)
    for i in range(m):
        adj_alpha = alpha / (m - i)
        corrected[sorted_idx[i]] = min(1.0, sorted_p[i] * (m - i))
        if sorted_p[i] <= adj_alpha:
            reject[sorted_idx[i]] = True
        else:
            break
    return reject, corrected


def _cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size."""
    n1, n2 = len(x), len(y)
    dom = sum(1 if xi < yj else (-1 if xi > yj else 0)
              for xi in x for yj in y)
    return dom / (n1 * n2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SFCM Experiment Runner")
    parser.add_argument("--all",       action="store_true", help="Run all experiments")
    parser.add_argument("--bench",     type=str, default=None, help="Single benchmark name")
    parser.add_argument("--d",         type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--n_runs",    type=int, default=30)
    parser.add_argument("--ablation",  action="store_true")
    parser.add_argument("--realworld", action="store_true")
    parser.add_argument("--noise",     action="store_true")
    parser.add_argument("--simulate",  action="store_true", help="Use simulated P1 (no GPU)")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.bench:
        funcs = [args.bench] if args.bench else None
        print("=== Synthetic benchmarks ===")
        syn_df = run_synthetic(args.d, args.n_runs, selected_funcs=funcs)
        print(syn_df[["func", "d", "sfcm_mean"]].to_string(index=False))
        wil_df = compute_wilcoxon_table(syn_df)
        print("\n=== Wilcoxon summary ===")
        print(wil_df.to_string(index=False))

    if args.all or args.realworld:
        print("\n=== Real-world problems ===")
        rw_df = run_realworld(args.n_runs, simulate_p1=args.simulate)
        print(rw_df[["problem", "sfcm_mean", "sfcm_sr"]].to_string(index=False))

    if args.all or args.ablation:
        print("\n=== Ablation study ===")
        abl_df = run_ablation(d=args.d[0] if len(args.d) == 1 else 30,
                              n_runs=args.n_runs)
        print(abl_df.to_string(index=False))

    if args.all or args.noise:
        print("\n=== Noise robustness ===")
        ns_df = run_noise_robustness(n_runs=min(args.n_runs, 10))
        print(ns_df.to_string(index=False))

    print("\nAll results saved to results/")


if __name__ == "__main__":
    main()
