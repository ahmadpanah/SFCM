"""
benchmarks/functions.py
=======================
All 12 synthetic benchmark functions used in the paper (§4.2.1).

Unimodal   : Sphere, Rosenbrock, Schwefel_2_2
Multimodal : Rastrigin, Griewank, Ackley, Schwefel, Levy, Michalewicz
CEC        : CEC_F1, CEC_F2, CEC_F3

Each function is callable:  f(x) -> float,  domain [-bound, bound]^d
All are to be MINIMISED.  Global optimum f* = 0 unless noted.

CEC provenance (important — see benchmarks/cec2017/VENDORED.md)
---------------------------------------------------------------
The paper (§4.2.1) cites "shifted/rotated hybrid and composite functions from the
CEC 2017 suite (F1: Hybrid Function 1, F2: Hybrid Function 2, F3: Composition
Function 1, reported in shifted error form f(x) − f*)".  In the official CEC 2017
numbering (Awad et al. 2016), Hybrid Function 1 = **f11**, Hybrid Function 2 = **f12**
and Composition Function 1 = **f21** (f1–f3 in that suite are unimodal, so the
paper's "CEC 2017" labels map to f11/f12/f21 and are reported here as the
manuscript names CEC_F1/CEC_F2/CEC_F3 for table compatibility).

get_benchmark_suite(d) uses by default the **official CEC 2017 implementation**
(vendored NumPy port benchmarks/cec2017, adapted from Awad's original C code, with
the official shift/rotation/shuffle data).  Values returned are the *official*
function values; BenchmarkFunction.optimum carries the known global minimum
(f11* = 1100, f12* = 1200, f21* = 2100), so downstream code subtracting the
optimum reports the manuscript's shifted-error form f(x) − f*.

The earlier hand-written "CEC-2017-style" mixes in this file are retained ONLY as
evidence of the original simplified versions; they are clearly suffixed
" (simplified)" and are served by get_simplified_cec_suite(d).  They must NOT be
used for results claimed to follow the CEC 2017 suite.

Supported official dimensionality: f11/f12 support D ∈ {10, 30, 50, 100} and f21
supports D ∈ {10, 30, 50}; the paper uses d ∈ {10, 30, 50} — all supported.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchmarkFunction:
    name: str
    func: Callable[[np.ndarray], float]
    bounds: float          # symmetric: [-bounds, bounds]^d
    optimum: float = 0.0   # known global minimum value


# ---------------------------------------------------------------------------
# Unimodal
# ---------------------------------------------------------------------------

def sphere(x: np.ndarray) -> float:
    """f(x) = Σ xᵢ²    f* = 0 at origin"""
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """f(x) = Σ [100(xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²]    f* = 0 at (1,…,1)"""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def schwefel_2_2(x: np.ndarray) -> float:
    """f(x) = Σ|xᵢ| + Π|xᵢ|    f* = 0 at origin"""
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))


# ---------------------------------------------------------------------------
# Multimodal
# ---------------------------------------------------------------------------

def rastrigin(x: np.ndarray) -> float:
    """f(x) = 10d + Σ[xᵢ² − 10cos(2πxᵢ)]    f* = 0 at origin"""
    d = len(x)
    return float(10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def griewank(x: np.ndarray) -> float:
    """f* = 0 at origin"""
    d = len(x)
    sum_sq = np.sum(x ** 2) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return float(1.0 + sum_sq - prod_cos)


def ackley(x: np.ndarray) -> float:
    """f* = 0 at origin"""
    d = len(x)
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    t1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    t2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return float(t1 + t2 + a + np.e)


def schwefel(x: np.ndarray) -> float:
    """
    Schwefel 2.26  f(x) = 418.9829d − Σ xᵢ sin(√|xᵢ|)
    f* ≈ 0 at x* ≈ (420.97, …, 420.97)
    """
    d = len(x)
    return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def levy(x: np.ndarray) -> float:
    """f* = 0 at (1,…,1)"""
    w = 1.0 + (x - 1.0) / 4.0
    t1 = np.sin(np.pi * w[0]) ** 2
    t2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    t3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    return float(t1 + t2 + t3)


def michalewicz(x: np.ndarray) -> float:
    """
    f(x) = −Σ sin(xᵢ) · sin²ᵐ(i·xᵢ²/π),  m=10
    Minimised (negated): f* ≈ −9.66 for d=10 in [0, π]^d
    Domain: [0, π]^d (bounds = π in BenchmarkFunction)
    """
    m = 10
    i = np.arange(1, len(x) + 1)
    return float(-np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m)))


# ---------------------------------------------------------------------------
# Official CEC 2017 functions (vendored NumPy port — see VENDORED.md)
# ---------------------------------------------------------------------------

class _OfficialCEC2017:
    """
    Wrapper around the official CEC 2017 function of a given number, evaluated
    for a single decision vector and returned as a plain float.

    Returns the RAW official value f(x) (minimum f* at the shifted optimum);
    BenchmarkFunction.optimum carries f* so that reporting code obtains the
    shifted-error form f(x) − f* used in the manuscript.
    """

    def __init__(self, fn_number: int, f_star: float):
        """fn_number: 1-indexed official CEC 2017 function (11, 12, 21, ...)."""
        self.fn_number = fn_number
        self.f_star = f_star
        self._official = _CEC2017_FNS[fn_number]

    def __call__(self, x: np.ndarray) -> float:
        x_arr = np.atleast_2d(np.asarray(x, dtype=float))
        return float(self._official(x_arr)[0])


# Locate the vendored package. It lives at benchmarks/cec2017, but the repo may
# also be used with a pip-installed cec2017; prefer the vendored copy.
def _load_official_cec_functions() -> dict:
    """Import {11: f11, 12: f12, 21: f21} from the vendored implementation."""
    try:
        from .cec2017 import hybrid as _hybrid, composition as _composition  # vendored
    except ImportError:            # pragma: no cover - fallback
        from cec2017 import hybrid as _hybrid, composition as _composition  # pip pkg
    return {11: _hybrid.f11, 12: _hybrid.f12, 21: _composition.f21}


_CEC2017_FNS = _load_official_cec_functions()

# Manuscript-name -> official CEC 2017 mapping (see module docstring)
CEC2017_MAPPING = {
    "CEC_F1": dict(fn_number=11, label="CEC 2017 F11 (Hybrid Function 1)", f_star=1100.0),
    "CEC_F2": dict(fn_number=12, label="CEC 2017 F12 (Hybrid Function 2)", f_star=1200.0),
    "CEC_F3": dict(fn_number=21, label="CEC 2017 F21 (Composition Function 1)", f_star=2100.0),
}


# ---------------------------------------------------------------------------
# CEC-2017-style hybrid / composite (simplified reproducible versions)
# ---------------------------------------------------------------------------

def _make_rotation(d: int, seed: int) -> np.ndarray:
    """Reproducible random orthogonal rotation matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(M)
    return Q


def _make_shift(d: int, bounds: float, seed: int) -> np.ndarray:
    """Fixed shift vector within bounds."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.4 * bounds, 0.4 * bounds, size=d)


class _CEC_F1:
    """
    CEC F1 — Hybrid function (CEC 2017 style).
    Composition of Rastrigin (50%) and Schwefel (50%) on rotated, shifted subspaces.
    """
    def __init__(self, d: int):
        self.d = d
        self.R1 = _make_rotation(d, seed=101)
        self.R2 = _make_rotation(d, seed=102)
        self.o1 = _make_shift(d, 100.0, seed=201)
        self.o2 = _make_shift(d, 100.0, seed=202)
        self.optimum = 0.0

    def __call__(self, x: np.ndarray) -> float:
        z1 = self.R1 @ (x - self.o1)
        z2 = self.R2 @ (x - self.o2)
        return 0.5 * rastrigin(z1 / 5.12 * 100) + 0.5 * schwefel(z2 / 100 * 500)


class _CEC_F2:
    """
    CEC F2 — Hybrid function.
    Composition of Ackley (40%), Rosenbrock (30%), Griewank (30%)
    on rotated, shifted subspaces.
    """
    def __init__(self, d: int):
        self.d = d
        self.R = [_make_rotation(d, seed=110 + i) for i in range(3)]
        self.o = [_make_shift(d, 32.0, seed=210 + i) for i in range(3)]
        self.optimum = 0.0

    def __call__(self, x: np.ndarray) -> float:
        z = [(self.R[i] @ (x - self.o[i])) for i in range(3)]
        return (
            0.4 * ackley(z[0] / 32.768 * 5.12)
            + 0.3 * rosenbrock(z[1] / 32.768 * 5.12 + 1)
            + 0.3 * griewank(z[2] / 32.768 * 600)
        )


class _CEC_F3:
    """
    CEC F3 — Composite function.
    Weighted composition of 5 basic functions with different scales/rotations.
    """
    def __init__(self, d: int):
        self.d = d
        self.R = [_make_rotation(d, seed=120 + i) for i in range(5)]
        self.o = [_make_shift(d, 100.0, seed=220 + i) for i in range(5)]
        self.sigma = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.lam = [1.0, 10.0, 1e-2, 1.0, 5e-4]
        self.funcs = [rastrigin, griewank, schwefel, ackley, sphere]
        self.bias = [0.0, 100.0, 200.0, 300.0, 400.0]
        self.optimum = 0.0

    def __call__(self, x: np.ndarray) -> float:
        w = np.array([
            np.exp(-np.sum((x - self.o[i]) ** 2) / (2.0 * self.d * self.sigma[i] ** 2))
            for i in range(5)
        ])
        w_sum = w.sum() if w.sum() > 1e-12 else 1.0
        w /= w_sum
        vals = np.array([
            self.lam[i] * self.funcs[i](self.R[i] @ (x - self.o[i])) + self.bias[i]
            for i in range(5)
        ])
        return float(np.dot(w, vals))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_benchmark_suite(
    d: int,
    use_official_cec: bool = True,
) -> list[BenchmarkFunction]:
    """
    Return all 12 benchmark functions configured for dimension d.
    Budgets are set externally (see configs/sfcm_default.yaml).

    use_official_cec=True  → the paper-experiment default: the official CEC 2017
                             functions f11/f12/f21 via the vendored port
                             (benchmarks/cec2017 — official shift/rotation data).
    use_official_cec=False → the three historical hand-written "CEC-2017-style"
                             mixes (names suffixed " (simplified)"); kept only for
                             backward-reference and must not be cited as CEC 2017.
    """
    if use_official_cec:
        cec_entries = []
        for name, spec in CEC2017_MAPPING.items():
            wrapper = _OfficialCEC2017(spec["fn_number"], spec["f_star"])
            # search range for f11/f12/f21 is [-100, 100]^D (Awad et al. 2016)
            cec_entries.append(BenchmarkFunction(name, wrapper, bounds=100.0,
                                                 optimum=spec["f_star"]))
    else:
        cec_f1 = _CEC_F1(d)
        cec_f2 = _CEC_F2(d)
        cec_f3 = _CEC_F3(d)
        cec_entries = [
            BenchmarkFunction("CEC_F1 (simplified)", cec_f1, bounds=100.0, optimum=0.0),
            BenchmarkFunction("CEC_F2 (simplified)", cec_f2, bounds=32.768, optimum=0.0),
            BenchmarkFunction("CEC_F3 (simplified)", cec_f3, bounds=100.0, optimum=0.0),
        ]

    return [
        # Unimodal
        BenchmarkFunction("Sphere",       sphere,       bounds=100.0,  optimum=0.0),
        BenchmarkFunction("Rosenbrock",   rosenbrock,   bounds=30.0,   optimum=0.0),
        BenchmarkFunction("Schwefel_2_2", schwefel_2_2, bounds=10.0,   optimum=0.0),
        # Multimodal
        BenchmarkFunction("Rastrigin",    rastrigin,    bounds=5.12,   optimum=0.0),
        BenchmarkFunction("Griewank",     griewank,     bounds=600.0,  optimum=0.0),
        BenchmarkFunction("Ackley",       ackley,       bounds=32.768, optimum=0.0),
        BenchmarkFunction("Schwefel",     schwefel,     bounds=500.0,  optimum=0.0),
        BenchmarkFunction("Levy",         levy,         bounds=10.0,   optimum=0.0),
        BenchmarkFunction("Michalewicz",  michalewicz,  bounds=np.pi,  optimum=-9.66 if d == 10 else -d * 0.966),
        # CEC 2017 hybrid/composite (official or historical simplified versions)
        *cec_entries,
    ]


def get_simplified_cec_suite(d: int) -> list[BenchmarkFunction]:
    """
    Return only the three historical hand-written "CEC-2017-style" mixes with
    explicit " (simplified)" name suffixes. These are NOT the official CEC 2017
    functions and any results from them must never be cited as official CEC 2017
    results. Provided for transparency about the original repository state.
    """
    return [f for f in get_benchmark_suite(d, use_official_cec=False)
            if f.name.endswith(" (simplified)")]
