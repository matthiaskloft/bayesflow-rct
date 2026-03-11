"""
ANCOVA 2-Arms Continuous Outcome: Re-export facade.

All implementations have been split into focused modules:
- config.py     -- ANCOVAConfig dataclass + build_networks()
- simulator.py  -- prior, likelihood, meta, factory functions
- adapter.py    -- adapter specification
- validation.py -- condition grids, simulate/infer callables, validation loop
- training.py   -- Optuna objective, threshold training helpers
- metadata.py   -- model metadata utilities

This file re-exports the public API for backward compatibility.
"""

# Config
# Adapter
from bayesflow_rct.models.ancova.adapter import (  # noqa: F401
    create_ancova_adapter,
    get_ancova_adapter_spec,
)
from bayesflow_rct.models.ancova.config import (  # noqa: F401
    ANCOVAConfig,
    build_networks,
    hpo_params_to_config,
)

# Metadata
from bayesflow_rct.models.ancova.metadata import (  # noqa: F401
    get_model_metadata,
    save_model_with_metadata,
)

# Simulator
from bayesflow_rct.models.ancova.simulator import (  # noqa: F401
    create_likelihood_fn,
    create_meta_fn,
    create_prior_fn,
    create_simulator,
    likelihood,
    meta,
    prior,
    simulate_cond_batch,
)

# Training
from bayesflow_rct.models.ancova.training import (  # noqa: F401
    create_ancova_objective,
    create_ancova_training_functions,
    create_ancova_workflow_components,
)

# Validation
from bayesflow_rct.models.ancova.validation import (  # noqa: F401
    build_validation_dataset,
    create_validation_grid,
    make_condition_infer_fn,
    make_infer_fn,
    make_simulate_fn,
    run_condition_grid_validation,
)


# ---------------------------------------------------------------------------
# Deprecation shims for removed config classes
# ---------------------------------------------------------------------------
def __getattr__(name: str):
    """Provide helpful error for removed config classes."""
    _removed = {
        "PriorConfig", "MetaConfig", "SummaryNetworkConfig",
        "InferenceNetworkConfig", "TrainingConfig", "WorkflowConfig",
    }
    if name in _removed:
        raise ImportError(
            f"{name} has been removed. Use the flat ANCOVAConfig instead.\n"
            f"  Before: config.workflow.inference_network.network_type\n"
            f"  After:  config.inference_network_type\n"
            f"  See: from bayesflow_rct.models.ancova.config import ANCOVAConfig"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config
    "ANCOVAConfig",
    "build_networks",
    "hpo_params_to_config",
    # Simulator
    "prior",
    "likelihood",
    "meta",
    "simulate_cond_batch",
    "create_prior_fn",
    "create_likelihood_fn",
    "create_meta_fn",
    "create_simulator",
    # Adapter
    "get_ancova_adapter_spec",
    "create_ancova_adapter",
    # Validation
    "create_validation_grid",
    "make_simulate_fn",
    "make_condition_infer_fn",
    "make_infer_fn",
    "run_condition_grid_validation",
    "build_validation_dataset",
    # Training
    "create_ancova_workflow_components",
    "create_ancova_objective",
    "create_ancova_training_functions",
    # Metadata
    "get_model_metadata",
    "save_model_with_metadata",
]
