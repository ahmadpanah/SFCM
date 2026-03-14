"""
SFCM: Surrogate-Feedback Co-evolutionary Metaheuristic
=======================================================
Public API
"""

from .core import SFCM, DEFAULT_HPARAMS
from .surrogate import EnsembleMLP, SurrogatePopulation, SurrogateConfig
from .operators import (
    latin_hypercube_init,
    trust_score,
    noise_aware_score,
    misprediction_density,
    de_rand_1_bin,
)

__version__ = "1.0.0"
__all__ = [
    "SFCM",
    "DEFAULT_HPARAMS",
    "EnsembleMLP",
    "SurrogatePopulation",
    "SurrogateConfig",
    "latin_hypercube_init",
    "trust_score",
    "noise_aware_score",
    "misprediction_density",
    "de_rand_1_bin",
]
