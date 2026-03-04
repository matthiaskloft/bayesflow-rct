"""RCT Bayesian Power Training using Neural Posterior Estimation."""

from importlib.metadata import PackageNotFoundError, version

from bayesflow_hpo import (
    AdapterSpec,
    CompositeSearchSpace,
    CouplingFlowSpace,
    DeepSetSpace,
    GenericObjective,
    PriorStandardize,
    TrainingSpace,
    ValidationDataset,
    WorkflowBuildConfig,
    build_inference_network,
    build_summary_network,
    build_workflow,
    create_adapter,
    create_study,
    get_workflow_metadata,
    load_workflow_with_metadata,
    optimize,
    run_validation_pipeline as run_hpo_validation_pipeline,
    save_workflow_with_metadata,
)

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
from bayesflow_rct.models.ancova.hpo import (
    get_or_create_validation_data,
    run_ancova_hpo,
)
from bayesflow_rct.models.ancova.model import (
    ANCOVAConfig,
    InferenceNetworkConfig,
    MetaConfig,
    PriorConfig,
    SummaryNetworkConfig,
    TrainingConfig,
    WorkflowConfig,
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
    "optimize",
    "create_study",
    "GenericObjective",
    "CompositeSearchSpace",
    "CouplingFlowSpace",
    "DeepSetSpace",
    "TrainingSpace",
    "ValidationDataset",
    "run_hpo_validation_pipeline",
    "loguniform_int",
    "loguniform_float",
    "sample_t_or_normal",
    # ANCOVA
    "ANCOVAConfig",
    "PriorConfig",
    "MetaConfig",
    "create_ancova_workflow_components",
    "get_or_create_validation_data",
    "run_ancova_hpo",
    # Plotting
    "plot_coverage_diff",
]
