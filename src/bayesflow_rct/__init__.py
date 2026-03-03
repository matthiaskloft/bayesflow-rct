"""RCT Bayesian Power Training using Neural Posterior Estimation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bayesflow-rct")
except PackageNotFoundError:
    __version__ = "0.1.0"  # fallback for uninstalled (e.g. raw checkout)

# Core infrastructure (generic, reusable)
from bayesflow_rct.core.infrastructure import (
    AdapterSpec,
    InferenceNetworkConfig,
    SummaryNetworkConfig,
    TrainingConfig,
    WorkflowConfig,
    build_inference_network,
    build_summary_network,
    build_workflow,
    load_workflow_with_metadata,
    save_workflow_with_metadata,
)
from bayesflow_rct.core.optimization import (
    HyperparameterSpace,
    create_optimization_objective,
    create_study,
)
from bayesflow_rct.core.utils import (
    loguniform_float,
    loguniform_int,
    sample_t_or_normal,
)
from bayesflow_rct.core.validation import (
    run_validation_pipeline,
)

# ANCOVA model
from bayesflow_rct.models.ancova.model import (
    ANCOVAConfig,
    MetaConfig,
    PriorConfig,
    create_ancova_workflow_components,
)

# Plotting
from bayesflow_rct.plotting.diagnostics import (
    plot_coverage_diff,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "SummaryNetworkConfig",
    "InferenceNetworkConfig",
    "TrainingConfig",
    "WorkflowConfig",
    "AdapterSpec",
    "build_summary_network",
    "build_inference_network",
    "build_workflow",
    "save_workflow_with_metadata",
    "load_workflow_with_metadata",
    "create_study",
    "HyperparameterSpace",
    "create_optimization_objective",
    "run_validation_pipeline",
    "loguniform_int",
    "loguniform_float",
    "sample_t_or_normal",
    # ANCOVA
    "ANCOVAConfig",
    "PriorConfig",
    "MetaConfig",
    "create_ancova_workflow_components",
    # Plotting
    "plot_coverage_diff",
]
