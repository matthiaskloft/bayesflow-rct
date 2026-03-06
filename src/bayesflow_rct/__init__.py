"""RCT Bayesian Power Training using Neural Posterior Estimation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bayesflow-rct")
except PackageNotFoundError:
    __version__ = "0.1.0"  # fallback for uninstalled (e.g. raw checkout)

from bayesflow_rct.core.utils import (
    loguniform_float,
    loguniform_int,
    sample_t_or_normal,
)

# ANCOVA model
from bayesflow_rct.models.ancova.config import ANCOVAConfig
from bayesflow_rct.models.ancova.hpo import (
    get_or_create_validation_data,
    run_ancova_hpo,
)
from bayesflow_rct.models.ancova.training import create_ancova_workflow_components

# Plotting
from bayesflow_rct.plotting.diagnostics import plot_coverage_diff

__all__ = [
    # Version
    "__version__",
    # Core utils
    "loguniform_int",
    "loguniform_float",
    "sample_t_or_normal",
    # ANCOVA model
    "ANCOVAConfig",
    "create_ancova_workflow_components",
    "get_or_create_validation_data",
    "run_ancova_hpo",
    # Plotting
    "plot_coverage_diff",
]
