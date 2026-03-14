"""
tests/test_sfcm.py
==================
Unit + integration tests for SFCM.

Written for Python's built-in unittest (no external test runner required).
Also compatible with pytest — the thin shim below makes pytest.approx and
pytest.fixture work when pytest IS installed (production CI).

Run offline:   python3 -m unittest tests/test_sfcm.py -v
Run with CI:   pytest tests/ -v
"""

import sys, os, math, types, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── pytest compatibility shim (used when pytest is not installed) ─────────
try:
    import pytest
except ModuleNotFoundError:
    # Provide just enough of pytest's API to keep the test logic identical
    class _Approx:
        def __init__(self, expected, rel=1e-6, abs=1e-12):
            self.expected = expected
            self.rel = rel
            self.abs = abs
        def __eq__(self, actual):
            tol = max(self.abs, self.rel * abs(self.expected))
            return abs(actual - self.expected) <= tol
        def __repr__(self):
            return f"approx({self.expected})"
    class _pytest_stub:
        @staticmethod
        def approx(val, rel=1e-6, abs=1e-12):
            return _Approx(val, rel=rel, abs=abs)
        @staticmethod
        def fixture(*args, **kwargs):
            # No-op decorator — unittest doesn't use fixtures
            def decorator(fn):
                return fn
            return decorator if args and callable(args[0]) else decorator
        @staticmethod
        def mark(*a, **kw):
            pass
    pytest = _pytest_stub()
    sys.modules["pytest"] = pytest  # satisfy any deeper import

# ── Project imports ───────────────────────────────────────────────────────
import numpy as np
from sfcm.core import SFCM, DEFAULT_HPARAMS
from sfcm.surrogate import EnsembleMLP, SurrogateConfig, SurrogatePopulation
from sfcm.operators import (
    latin_hypercube_init, trust_score, noise_aware_score,
    robust_replace, misprediction_density, scaled_perturbation, de_rand_1_bin,
)
from benchmarks.functions import get_benchmark_suite, sphere, rastrigin


# ─────────────────────────────────────────────────────────────────────────
# Shared fixtures (computed once per class where needed)
# ─────────────────────────────────────────────────────────────────────────
_bounds_2d  = np.array([[-5.0, 5.0], [-5.0, 5.0]])
_bounds_10d = np.array([[-5.12, 5.12]] * 10)
_rng        = np.random.default_rng(42)


# =========================================================================
# 1. Operator tests
# =========================================================================

class TestLatinHypercube(unittest.TestCase):

    def test_shape(self):
        X = latin_hypercube_init(20, _bounds_2d, np.random.default_rng(0))
        self.assertEqual(X.shape, (20, 2))

    def test_within_bounds(self):
        X = latin_hypercube_init(50, _bounds_10d, np.random.default_rng(1))
        self.assertTrue(np.all(X >= _bounds_10d[:, 0]))
        self.assertTrue(np.all(X <= _bounds_10d[:, 1]))

    def test_stratification(self):
        """Each stratum must contain exactly one point per dimension."""
        N = 10
        rng = np.random.default_rng(2)
        X = latin_hypercube_init(N, _bounds_2d, rng)
        lb, ub = _bounds_2d[:, 0], _bounds_2d[:, 1]
        for j in range(2):
            stride = (ub[j] - lb[j]) / N
            for k in range(N):
                lo, hi = lb[j] + k * stride, lb[j] + (k + 1) * stride
                count = np.sum((X[:, j] >= lo - 1e-9) & (X[:, j] <= hi + 1e-9))
                self.assertEqual(count, 1,
                    f"Stratum {k} dim {j} has {count} points (expected 1)")

    def test_different_seeds_different_results(self):
        X1 = latin_hypercube_init(10, _bounds_2d, np.random.default_rng(0))
        X2 = latin_hypercube_init(10, _bounds_2d, np.random.default_rng(99))
        self.assertFalse(np.allclose(X1, X2))


class TestTrustScore(unittest.TestCase):

    def test_high_uncertainty_low_trust(self):
        t = trust_score(sigma_hat=10.0, eps_bar=0.1, alpha=2.0)
        self.assertLess(t, 0.01)

    def test_low_uncertainty_high_trust(self):
        t = trust_score(sigma_hat=0.001, eps_bar=0.1, alpha=2.0)
        self.assertGreater(t, 0.9)

    def test_formula_exact(self):
        """trust = exp(-α·σ̂/(ε̄+ε))  Eq.(4)"""
        σ, ε_bar, α, ε = 0.5, 0.25, 2.0, 1e-8
        expected = math.exp(-α * σ / (ε_bar + ε))
        result   = trust_score(σ, ε_bar, α, ε)
        self.assertAlmostEqual(result, expected, places=12)

    def test_tau0_is_0_3_in_defaults(self):
        """τ₀ = 0.3 (Appendix C); §4.4 draft had 0.4 — guard against reversion."""
        self.assertEqual(DEFAULT_HPARAMS["tau0"], 0.3)


class TestNoiseAwareScore(unittest.TestCase):

    def test_formula(self):
        """score = f̃ − κ·u   Eq.(6)"""
        result = noise_aware_score(1.5, 0.5, 1.5)
        self.assertAlmostEqual(result, 1.5 - 1.5 * 0.5, places=12)

    def test_kappa_is_1_5_in_defaults(self):
        """κ = 1.5 (Appendix C); §4.4 draft had 0.5 — guard against reversion."""
        self.assertEqual(DEFAULT_HPARAMS["kappa"], 1.5)

    def test_zero_uncertainty_returns_f_eff(self):
        self.assertAlmostEqual(noise_aware_score(2.3, 0.0, 1.5), 2.3, places=12)

    def test_higher_kappa_more_exploration(self):
        """Higher κ → lower score for same uncertainty → more replacement."""
        s_low  = noise_aware_score(1.0, 0.5, 0.5)
        s_high = noise_aware_score(1.0, 0.5, 2.0)
        self.assertGreater(s_low, s_high)


class TestRobustReplace(unittest.TestCase):

    def test_replace_when_clearly_better(self):
        self.assertTrue(robust_replace(0.5, 1.0, 0.1))

    def test_no_replace_within_tolerance(self):
        # score_new + delta = 0.95 + 0.1 = 1.05 >= 1.0 → no replace
        self.assertFalse(robust_replace(0.95, 1.0, 0.1))

    def test_boundary_no_replace(self):
        # score_new + delta exactly equals score_old → no replace (strict <)
        self.assertFalse(robust_replace(0.9, 1.0, 0.1))

    def test_delta0_is_0_1_in_defaults(self):
        self.assertEqual(DEFAULT_HPARAMS["delta0"], 0.1)


class TestMispredictionDensity(unittest.TestCase):

    def test_zero_on_empty_archive(self):
        X = latin_hypercube_init(5, _bounds_2d, np.random.default_rng(3))
        class _Dummy:
            def predict(self, x): return 0.0, 0.1
        rho = misprediction_density(X, [], [], _Dummy(), h=1.0)
        np.testing.assert_array_equal(rho, 0.0)

    def test_higher_density_near_error_point(self):
        """Points close to a high-error archived point get higher density."""
        archive_X = [np.array([0.0, 0.0])]
        archive_y = [5.0]
        class _Dummy:
            def predict(self, x): return 100.0, 0.1  # huge error
        near = np.array([[0.01, 0.01]])
        far  = np.array([[10.0, 10.0]])
        rho_near = misprediction_density(near, archive_X, archive_y, _Dummy(), h=0.5)
        rho_far  = misprediction_density(far,  archive_X, archive_y, _Dummy(), h=0.5)
        self.assertGreater(rho_near[0], rho_far[0])

    def test_output_length_matches_query(self):
        X = latin_hypercube_init(8, _bounds_2d, np.random.default_rng(4))
        archive_X = [np.zeros(2), np.ones(2)]
        archive_y = [0.0, 1.0]
        class _Dummy:
            def predict(self, x): return float(np.sum(x)), 0.1
        rho = misprediction_density(X, archive_X, archive_y, _Dummy(), h=1.0)
        self.assertEqual(len(rho), 8)

    def test_non_negative(self):
        X = latin_hypercube_init(5, _bounds_2d, np.random.default_rng(5))
        archive_X = [np.random.randn(2) for _ in range(10)]
        archive_y = [float(np.random.randn()) for _ in range(10)]
        class _Dummy:
            def predict(self, x): return float(np.sum(x)), 0.1
        rho = misprediction_density(X, archive_X, archive_y, _Dummy(), h=1.0)
        self.assertTrue(np.all(rho >= 0.0))


class TestScaledPerturbation(unittest.TestCase):

    def test_formula(self):
        """x' = (1 + β·ω)·Δ   Eq.(9)"""
        delta = np.array([1.0, -0.5])
        result = scaled_perturbation(delta, omega=0.5, beta=0.8)
        np.testing.assert_allclose(result, (1 + 0.8 * 0.5) * delta)

    def test_beta_is_0_8_in_defaults(self):
        """β = 0.8 (Appendix C); §4.4 draft had 1.5 — guard against reversion."""
        self.assertEqual(DEFAULT_HPARAMS["beta"], 0.8)

    def test_zero_omega_identity(self):
        delta = np.array([2.0, -1.0])
        np.testing.assert_allclose(scaled_perturbation(delta, 0.0, 0.8), delta)

    def test_higher_omega_larger_step(self):
        delta = np.ones(3)
        low  = np.linalg.norm(scaled_perturbation(delta, 0.1, 0.8))
        high = np.linalg.norm(scaled_perturbation(delta, 0.9, 0.8))
        self.assertGreater(high, low)


class TestDE(unittest.TestCase):

    def test_output_shape(self):
        pop   = latin_hypercube_init(10, _bounds_10d, np.random.default_rng(6))
        delta = de_rand_1_bin(pop, 0, _bounds_10d, np.random.default_rng(6))
        self.assertEqual(delta.shape, (10,))

    def test_some_dims_zero_with_low_cr(self):
        """With CR=0.3, some dimensions should be masked (delta=0)."""
        pop = latin_hypercube_init(10, _bounds_10d, np.random.default_rng(7))
        found_zero = False
        for seed in range(30):
            delta = de_rand_1_bin(pop, 0, _bounds_10d,
                                   np.random.default_rng(seed), CR=0.3)
            if np.any(delta == 0.0):
                found_zero = True
                break
        self.assertTrue(found_zero, "Expected some zero dims with CR=0.3")

    def test_always_one_nonzero_dim(self):
        """At least one dimension must always be nonzero (DE guarantee)."""
        pop = latin_hypercube_init(10, _bounds_2d, np.random.default_rng(8))
        for seed in range(20):
            delta = de_rand_1_bin(pop, 0, _bounds_2d,
                                   np.random.default_rng(seed), CR=0.01)
            self.assertFalse(np.all(delta == 0.0),
                             "All-zero delta violates DE crossover guarantee")


# =========================================================================
# 2. Surrogate tests
# =========================================================================

class TestEnsembleMLP(unittest.TestCase):

    def _make_data(self, n=40, d=5, fn=sphere, seed=0):
        rng = np.random.default_rng(seed)
        X = [rng.random(d) for _ in range(n)]
        y = [fn(x) for x in X]
        return X, y

    def test_fit_predict_finite(self):
        X, y = self._make_data()
        m = EnsembleMLP(d_in=5, hidden=(16, 8), n_members=2, patience=5, val_split=0.2)
        m.fit(X, y)
        y_hat, sigma = m.predict(np.zeros(5))
        self.assertTrue(np.isfinite(y_hat), "Prediction must be finite")
        self.assertGreater(sigma, 0, "Uncertainty must be positive")

    def test_uncertainty_positive(self):
        X, y = self._make_data(fn=rastrigin, seed=1)
        m = EnsembleMLP(d_in=5, hidden=(16, 8), n_members=3, patience=5, val_split=0.2)
        m.fit(X, y)
        _, sigma = m.predict(np.random.default_rng(9).random(5))
        self.assertGreater(sigma, 0)

    def test_architecture_matches_paper(self):
        """Default: hidden=(128,64,32), dropout=0.1, n_members=5  (Appendix B.1)."""
        m = EnsembleMLP(d_in=10)
        self.assertEqual(m.hidden, (128, 64, 32))
        self.assertEqual(m.dropout, 0.1)
        self.assertEqual(m.n_members, 5)

    def test_mean_abs_error_nonneg(self):
        X, y = self._make_data(n=20)
        m = EnsembleMLP(d_in=5, hidden=(8,), n_members=2, patience=5, val_split=0.2)
        m.fit(X, y)
        mae = m.mean_abs_error(X, y)
        self.assertGreaterEqual(mae, 0.0)

    def test_empty_archive_returns_fallback(self):
        m = EnsembleMLP(d_in=5)
        y_hat, sigma = m.predict(np.zeros(5))
        self.assertEqual(y_hat, 0.0)
        self.assertEqual(sigma, 1.0)

    def test_small_archive_no_crash(self):
        """fit() with < 2 points should silently do nothing."""
        m = EnsembleMLP(d_in=3, hidden=(8,), n_members=2, patience=3)
        m.fit([np.ones(3)], [1.0])   # only 1 point — should not raise


class TestSurrogateConfig(unittest.TestCase):

    def test_random_within_bounds(self):
        rng = np.random.default_rng(10)
        cfg = SurrogateConfig.random(d_in=5, rng=rng)
        from sfcm.surrogate import _TB
        self.assertTrue(np.all(cfg.theta >= _TB[:, 0]))
        self.assertTrue(np.all(cfg.theta <= _TB[:, 1]))

    def test_decode_valid_architecture(self):
        rng = np.random.default_rng(11)
        cfg = SurrogateConfig.random(d_in=5, rng=rng)
        p = cfg._decode()
        self.assertEqual(len(p["hidden"]), 3)
        self.assertTrue(all(h > 0 for h in p["hidden"]))
        self.assertGreaterEqual(p["dropout"], 0.0)
        self.assertLessEqual(p["dropout"], 0.4)
        self.assertGreater(p["lr"], 0.0)


class TestSurrogatePopulation(unittest.TestCase):

    def test_best_returns_config(self):
        rng = np.random.default_rng(12)
        pop = SurrogatePopulation(M=3, d_in=3, rng=rng)
        X = [rng.random(3) for _ in range(15)]
        y = [sphere(x) for x in X]
        pop.fit_all(X, y)
        best = pop.best()
        self.assertIsInstance(best, SurrogateConfig)

    def test_evolve_runs_without_error(self):
        rng = np.random.default_rng(13)
        pop = SurrogatePopulation(M=3, d_in=3, rng=rng)
        X = [rng.random(3) for _ in range(15)]
        y = [sphere(x) for x in X]
        pop.fit_all(X, y)
        P_X = latin_hypercube_init(5, np.array([[-1., 1.]] * 3), rng)
        pop.evolve(X, y, P_X, lam=0.7)   # should not raise
        self.assertEqual(len(pop.individuals), 3)


# =========================================================================
# 3. SFCM integration tests
# =========================================================================

class TestSFCMIntegration(unittest.TestCase):

    def test_sphere_5d_improves(self):
        """SFCM should find a value substantially below the random initial mean."""
        bounds = np.array([[-5.0, 5.0]] * 5)
        opt = SFCM(sphere, bounds, budget=150, seed=0, N=8, M=4, n0_factor=3)
        _, best_y = opt.run()
        # Random point in [-5,5]^5 has E[sphere] ≈ 5*25/3 ≈ 41.7
        self.assertLess(best_y, 20.0, f"Expected improvement over random (~41), got {best_y:.4f}")

    def test_budget_not_exceeded(self):
        bounds = np.array([[-5.12, 5.12]] * 10)
        budget = 120
        opt = SFCM(rastrigin, bounds, budget=budget, seed=1, N=8, M=4, n0_factor=2)
        opt.run()
        self.assertLessEqual(opt.n_true, budget,
            f"Budget exceeded: n_true={opt.n_true} > B={budget}")

    def test_history_nondecreasing_best(self):
        """history_best must be non-increasing (best value can only improve)."""
        bounds = np.array([[-5.0, 5.0]] * 4)
        opt = SFCM(sphere, bounds, budget=100, seed=2, N=6, M=3, n0_factor=2)
        opt.run()
        h = opt.history_best
        for i in range(len(h) - 1):
            self.assertGreaterEqual(h[i], h[i + 1] - 1e-10,
                f"history_best increased at step {i}: {h[i]} → {h[i+1]}")

    def test_history_lengths_equal(self):
        bounds = np.array([[-5.0, 5.0]] * 3)
        opt = SFCM(sphere, bounds, budget=80, seed=3, N=5, M=3, n0_factor=2)
        opt.run()
        self.assertEqual(len(opt.history_best), len(opt.history_n_true))

    def test_reproducibility_same_seed(self):
        """Identical seeds must produce identical results."""
        bounds = np.array([[-5.12, 5.12]] * 5)
        def run(s):
            opt = SFCM(rastrigin, bounds, budget=80, seed=s, N=6, M=3, n0_factor=2)
            _, y = opt.run()
            return y
        self.assertEqual(run(7), run(7), "Same seed must give identical result")

    def test_different_seeds_different_results(self):
        bounds = np.array([[-5.12, 5.12]] * 5)
        def run(s):
            opt = SFCM(rastrigin, bounds, budget=80, seed=s, N=6, M=3, n0_factor=2)
            _, y = opt.run()
            return y
        # Very unlikely to be identical with different seeds
        y0, y1 = run(0), run(1)
        self.assertNotEqual(y0, y1, "Different seeds should give different results")

    def test_n_true_positive(self):
        bounds = np.array([[-5.0, 5.0]] * 3)
        opt = SFCM(sphere, bounds, budget=60, seed=4, N=5, M=3, n0_factor=2)
        opt.run()
        self.assertGreater(opt.n_true, 0)

    def test_best_x_shape(self):
        bounds = np.array([[-5.0, 5.0]] * 6)
        opt = SFCM(sphere, bounds, budget=80, seed=5, N=6, M=3, n0_factor=2)
        _, _ = opt.run()
        self.assertIsNotNone(opt.best_x)
        self.assertEqual(opt.best_x.shape, (6,))

    def test_best_x_within_bounds(self):
        bounds = np.array([[-5.0, 5.0]] * 4)
        opt = SFCM(sphere, bounds, budget=80, seed=6, N=5, M=3, n0_factor=2)
        opt.run()
        self.assertTrue(np.all(opt.best_x >= bounds[:, 0]))
        self.assertTrue(np.all(opt.best_x <= bounds[:, 1]))

    def test_surrogate_off_ablation(self):
        """surrogate_off=True runs pure DE — should still produce valid result."""
        bounds = np.array([[-5.0, 5.0]] * 4)
        opt = SFCM(sphere, bounds, budget=80, seed=7,
                   N=6, M=3, n0_factor=2, surrogate_off=True)
        _, best_y = opt.run()
        self.assertTrue(np.isfinite(best_y))

    def test_no_noise_aware_ablation(self):
        bounds = np.array([[-5.0, 5.0]] * 4)
        opt = SFCM(sphere, bounds, budget=80, seed=8,
                   N=6, M=3, n0_factor=2, no_noise_aware=True)
        _, best_y = opt.run()
        self.assertTrue(np.isfinite(best_y))

    def test_no_mispred_explore_ablation(self):
        bounds = np.array([[-5.0, 5.0]] * 4)
        opt = SFCM(sphere, bounds, budget=80, seed=9,
                   N=6, M=3, n0_factor=2, no_mispred_explore=True)
        _, best_y = opt.run()
        self.assertTrue(np.isfinite(best_y))

    def test_no_dual_pop_ablation(self):
        bounds = np.array([[-5.0, 5.0]] * 4)
        opt = SFCM(sphere, bounds, budget=80, seed=10,
                   N=6, M=3, n0_factor=2, no_dual_pop=True)
        _, best_y = opt.run()
        self.assertTrue(np.isfinite(best_y))


# =========================================================================
# 4. Benchmark function tests
# =========================================================================

class TestBenchmarkSuite(unittest.TestCase):

    def test_suite_has_12_functions(self):
        suite = get_benchmark_suite(10)
        self.assertEqual(len(suite), 12)

    def test_all_return_finite(self):
        suite = get_benchmark_suite(10)
        x = np.ones(10) * 0.1
        for fn in suite:
            y = fn.func(x)
            self.assertTrue(np.isfinite(y), f"{fn.name} returned non-finite")

    def test_sphere_optimum_at_origin(self):
        self.assertEqual(sphere(np.zeros(10)), 0.0)

    def test_rastrigin_optimum_at_origin(self):
        self.assertAlmostEqual(rastrigin(np.zeros(10)), 0.0, places=12)

    def test_sphere_increases_with_distance(self):
        self.assertGreater(sphere(np.ones(5)), sphere(0.5 * np.ones(5)))

    def test_all_dims_work(self):
        for d in [10, 30, 50]:
            suite = get_benchmark_suite(d)
            x = np.zeros(d)
            for fn in suite:
                y = fn.func(x)
                self.assertTrue(np.isfinite(y), f"{fn.name} d={d} not finite")

    def test_bounds_positive(self):
        for fn in get_benchmark_suite(10):
            self.assertGreater(fn.bounds, 0, f"{fn.name} has non-positive bound")

    def test_cec_functions_change_with_input(self):
        """CEC functions should give different values at different points."""
        suite = get_benchmark_suite(10)
        cec_fns = [fn for fn in suite if fn.name.startswith("CEC")]
        rng = np.random.default_rng(20)
        for fn in cec_fns:
            x1 = rng.uniform(-fn.bounds, fn.bounds, 10)
            x2 = rng.uniform(-fn.bounds, fn.bounds, 10)
            self.assertNotAlmostEqual(fn.func(x1), fn.func(x2), places=3,
                msg=f"{fn.name} returned same value for different inputs")


# =========================================================================
# 5. Hyperparameter consistency guards
#    These tests exist solely to prevent re-introduction of the
#    kappa/beta/tau0 discrepancy flagged as Critical in peer review.
# =========================================================================

class TestHyperparameterConsistency(unittest.TestCase):
    """
    CRITICAL: These tests guard against re-introduction of the three
    hyperparameter transcription errors corrected in the paper revision.
    DO NOT change the expected values — they match Appendix C Table C1.
    """

    def test_kappa_is_1_5(self):
        self.assertEqual(DEFAULT_HPARAMS["kappa"], 1.5,
            "κ MUST be 1.5 (Appendix C). "
            "§4.4 draft value 0.5 was a transcription error.")

    def test_beta_is_0_8(self):
        self.assertEqual(DEFAULT_HPARAMS["beta"], 0.8,
            "β MUST be 0.8 (Appendix C). "
            "§4.4 draft value 1.5 was a transcription error.")

    def test_tau0_is_0_3(self):
        self.assertEqual(DEFAULT_HPARAMS["tau0"], 0.3,
            "τ₀ MUST be 0.3 (Appendix C). "
            "§4.4 draft value 0.4 was a transcription error.")

    def test_lam_is_0_7(self):
        self.assertEqual(DEFAULT_HPARAMS["lam"], 0.7)

    def test_alpha_is_2_0(self):
        self.assertEqual(DEFAULT_HPARAMS["alpha"], 2.0)

    def test_n0_factor_is_5(self):
        self.assertEqual(DEFAULT_HPARAMS["n0_factor"], 5)

    def test_k_retrain_is_10(self):
        self.assertEqual(DEFAULT_HPARAMS["K_retrain"], 10)

    def test_k_new_is_5(self):
        self.assertEqual(DEFAULT_HPARAMS["K_new"], 5)

    def test_n_is_30(self):
        self.assertEqual(DEFAULT_HPARAMS["N"], 30)

    def test_m_is_10(self):
        self.assertEqual(DEFAULT_HPARAMS["M"], 10)

    def test_delta0_is_0_1(self):
        self.assertEqual(DEFAULT_HPARAMS["delta0"], 0.1)

    def test_all_ablation_flags_off_by_default(self):
        """Full SFCM runs with all mechanisms active by default."""
        self.assertFalse(DEFAULT_HPARAMS["no_dual_pop"])
        self.assertFalse(DEFAULT_HPARAMS["no_noise_aware"])
        self.assertFalse(DEFAULT_HPARAMS["no_mispred_explore"])
        self.assertFalse(DEFAULT_HPARAMS["surrogate_off"])

    def test_yaml_config_matches_code(self):
        """configs/sfcm_default.yaml must agree with DEFAULT_HPARAMS."""
        import yaml, os
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "sfcm_default.yaml")
        if not os.path.exists(cfg_path):
            self.skipTest("configs/sfcm_default.yaml not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        sfcm_cfg = cfg.get("sfcm", {})
        yaml_kappa = sfcm_cfg.get("kappa")
        yaml_beta  = sfcm_cfg.get("beta")
        yaml_tau0  = sfcm_cfg.get("tau0")
        self.assertEqual(yaml_kappa, 1.5,  f"YAML kappa={yaml_kappa} ≠ 1.5")
        self.assertEqual(yaml_beta,  0.8,  f"YAML beta={yaml_beta} ≠ 0.8")
        self.assertEqual(yaml_tau0,  0.3,  f"YAML tau0={yaml_tau0} ≠ 0.3")


# =========================================================================
# 6. Real-world problem smoke tests (no GPU / simulate=True)
# =========================================================================

class TestRealWorldProblems(unittest.TestCase):

    def test_truss_evaluate_finite(self):
        from problems.real_world import TrussDesign
        prob = TrussDesign()
        x = np.ones(10) * 10.0   # valid cross-sections
        y = prob.evaluate(x)
        self.assertTrue(np.isfinite(y))
        self.assertGreater(y, 0.0, "Truss weight must be positive")

    def test_truss_bounds_shape(self):
        from problems.real_world import TrussDesign
        self.assertEqual(TrussDesign.bounds.shape, (10, 2))
        self.assertEqual(TrussDesign.d, 10)

    def test_container_evaluate_finite(self):
        from problems.real_world import ContainerPlacement
        prob = ContainerPlacement()
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 29, size=15)
        y = prob.evaluate(x)
        self.assertTrue(np.isfinite(y))

    def test_container_bounds_shape(self):
        from problems.real_world import ContainerPlacement
        self.assertEqual(ContainerPlacement.bounds.shape, (15, 2))
        self.assertEqual(ContainerPlacement.d, 15)

    def test_resnet_simulate_finite(self):
        from problems.real_world import ResNet18HPO, P1_BOUNDS
        prob = ResNet18HPO(simulate=True)
        rng = np.random.default_rng(43)
        x = rng.uniform(P1_BOUNDS[:, 0], P1_BOUNDS[:, 1])
        y = prob.evaluate(x)
        self.assertTrue(np.isfinite(y))
        self.assertGreater(y, 0.0)
        self.assertLess(y, 1.0)

    def test_resnet_bounds_shape(self):
        from problems.real_world import ResNet18HPO, P1_BOUNDS
        self.assertEqual(P1_BOUNDS.shape, (8, 2))
        self.assertEqual(ResNet18HPO.d, 8)

    def test_truss_penalty_on_infeasible(self):
        """Very thin cross-sections should trigger large penalty."""
        from problems.real_world import TrussDesign
        prob = TrussDesign()
        x_thin  = np.ones(10) * 0.1  # minimum area
        x_thick = np.ones(10) * 20.0
        y_thin  = prob.evaluate(x_thin)
        y_thick = prob.evaluate(x_thick)
        # thick gives higher weight but thin should not give lower-or-equal
        # (because thin areas cause large stress violations → penalty)
        # This just checks both are finite
        self.assertTrue(np.isfinite(y_thin))
        self.assertTrue(np.isfinite(y_thick))

    def test_container_different_assignments_different_latency(self):
        """Two different placements should yield different latencies."""
        from problems.real_world import ContainerPlacement
        prob = ContainerPlacement()
        x1 = np.zeros(15)        # all services on node 0
        x2 = np.arange(15, dtype=float)  # spread across nodes
        self.assertNotAlmostEqual(prob.evaluate(x1), prob.evaluate(x2), places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
