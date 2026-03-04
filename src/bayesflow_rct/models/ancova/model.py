"""
ANCOVA 2-Arms Continuous Outcome: Model-Specific Implementation

This module provides ANCOVA-specific implementations:
- Simulator functions (prior, likelihood, meta) for ANCOVA model
- Adapter specification for ANCOVA data structure
- Factory functions for creating ANCOVA workflows
- Validation pipeline helpers
- Model metadata utilities

Generic network/adaptation infrastructure is provided by bayesflow_hpo.
`bayesflow_rct.core.infrastructure` now only exposes a slim simulator helper.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import time
from typing import TYPE_CHECKING, Any

import bayesflow as bf
import bayesflow_hpo as hpo
import numpy as np
import pandas as pd
from bayesflow_hpo.builders import (
    AdapterSpec,
    WorkflowBuildConfig,
    build_inference_network,
    build_summary_network,
    build_workflow,
    create_adapter,
)
from bayesflow_hpo.results import (
    get_workflow_metadata,
    load_workflow_with_metadata,
    save_workflow_with_metadata,
)
from bayesflow_hpo.validation.inference import make_bayesflow_infer_fn as _make_bayesflow_infer_fn
from bayesflow_hpo.validation.metrics import aggregate_metrics, compute_batch_metrics

if TYPE_CHECKING:
    from bayesflow import Adapter, Simulator

from bayesflow_rct.core.infrastructure import (
    create_simulator as create_generic_simulator,
)
from bayesflow_rct.core.utils import loguniform_int, sample_t_or_normal

# =============================================================================
# ANCOVA-Specific Configuration Dataclasses
# =============================================================================


@dataclass
class PriorConfig:
    """ANCOVA-specific prior distribution parameters."""

    b_covariate_scale: float = 2.0  # Scale for b_covariate Normal distribution
    # Note: sigma is fixed at 1.0 in this model (not estimated)


@dataclass
class MetaConfig:
    """ANCOVA-specific meta-parameter sampling ranges for training."""

    n_min: int = 20
    n_max: int = 1000
    p_alloc_min: float = 0.5
    p_alloc_max: float = 0.9
    prior_df_min: int = 0  # 0 means Normal (df > 100 treated as Normal)
    prior_df_max: int = 30
    prior_df_alpha: float = 0.7  # Alpha for log-uniform sampling
    prior_scale_gamma_shape: float = 2.0
    prior_scale_gamma_scale: float = 1.0


@dataclass
class SummaryNetworkConfig:
    """Thin summary-network config shim retained for ANCOVA ergonomics."""

    summary_dim: int = 10
    depth: int = 3
    width: int = 64
    dropout: float = 0.05
    network_type: str = "DeepSet"


@dataclass
class InferenceNetworkConfig:
    """Thin inference-network config shim retained for ANCOVA ergonomics."""

    network_type: str = "FlowMatching"
    depth: int = 7
    hidden_sizes: tuple[int, ...] = (128, 128)
    widths: tuple[int, ...] = (128, 128, 128)
    use_optimal_transport: bool = False
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters used by ANCOVA wrappers."""

    initial_lr: float = 7e-4
    decay_rate: float = 0.85
    batch_size: int = 320
    epochs: int = 200
    batches_per_epoch: int = 50
    validation_sims: int = 1000
    early_stopping_patience: int = 10
    early_stopping_window: int = 5


@dataclass
class WorkflowConfig:
    """ANCOVA-facing workflow config shim mapped to bayesflow_hpo builders."""

    summary_network: SummaryNetworkConfig = field(default_factory=SummaryNetworkConfig)
    inference_network: InferenceNetworkConfig = field(
        default_factory=InferenceNetworkConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return {
            "summary_network": asdict(self.summary_network),
            "inference_network": asdict(self.inference_network),
            "training": asdict(self.training),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowConfig":
        return cls(
            summary_network=SummaryNetworkConfig(**d.get("summary_network", {})),
            inference_network=InferenceNetworkConfig(**d.get("inference_network", {})),
            training=TrainingConfig(**d.get("training", {})),
        )


def params_dict_to_workflow_config(params: dict[str, Any]) -> WorkflowConfig:
    """Convert flat hyperparameter dict (e.g. from Optuna) to `WorkflowConfig`."""
    network_type = params.get("network_type", "CouplingFlow")
    flow_depth = int(params.get("flow_depth", 7))
    flow_dropout = float(params.get("flow_dropout", 0.05))

    if network_type == "FlowMatching":
        flow_width = int(params.get("flow_width", 256))
        inference_config = InferenceNetworkConfig(
            network_type="FlowMatching",
            widths=(flow_width,) * flow_depth,
            dropout=flow_dropout,
            use_optimal_transport=bool(params.get("use_optimal_transport", False)),
        )
    else:
        flow_hidden = int(params.get("flow_hidden", 128))
        inference_config = InferenceNetworkConfig(
            network_type="CouplingFlow",
            depth=flow_depth,
            hidden_sizes=(flow_hidden,) * 2,
            dropout=flow_dropout,
        )

    return WorkflowConfig(
        summary_network=SummaryNetworkConfig(
            summary_dim=int(params.get("summary_dim", 10)),
            depth=int(params.get("deepset_depth", 3)),
            width=int(params.get("deepset_width", 64)),
            dropout=float(params.get("deepset_dropout", 0.05)),
        ),
        inference_network=inference_config,
        training=TrainingConfig(
            initial_lr=float(params.get("initial_lr", 7e-4)),
            batch_size=int(params.get("batch_size", 320)),
            decay_rate=float(params.get("decay_rate", 0.85)),
            epochs=int(params.get("epochs", 200)),
            batches_per_epoch=int(params.get("batches_per_epoch", 50)),
            validation_sims=int(params.get("validation_sims", 1000)),
            early_stopping_patience=int(params.get("early_stopping_patience", 10)),
            early_stopping_window=int(params.get("early_stopping_window", 5)),
        ),
    )


def _build_networks_from_workflow_config(
    workflow_config: WorkflowConfig,
) -> tuple[bf.networks.SummaryNetwork | None, bf.networks.InferenceNetwork]:
    """Build summary/inference networks via `bayesflow_hpo.builders` from shim config."""
    summary_space = hpo.DeepSetSpace()
    summary_params: dict[str, Any] = {
        "ds_summary_dim": int(workflow_config.summary_network.summary_dim),
        "ds_depth": int(workflow_config.summary_network.depth),
        "ds_width": int(workflow_config.summary_network.width),
        "ds_dropout": float(workflow_config.summary_network.dropout),
    }

    inference_cfg = workflow_config.inference_network
    if inference_cfg.network_type == "FlowMatching":
        inference_space = hpo.FlowMatchingSpace()
        fm_width = int(inference_cfg.widths[0]) if inference_cfg.widths else 128
        inference_params = {
            "fm_subnet_width": fm_width,
            "fm_subnet_depth": int(len(inference_cfg.widths) or 3),
            "fm_dropout": float(inference_cfg.dropout),
            "fm_activation": "mish",
            "fm_use_ot": bool(inference_cfg.use_optimal_transport),
        }
    elif inference_cfg.network_type == "CouplingFlow":
        inference_space = hpo.CouplingFlowSpace()
        cf_width = int(inference_cfg.hidden_sizes[0]) if inference_cfg.hidden_sizes else 128
        inference_params = {
            "cf_depth": int(inference_cfg.depth),
            "cf_subnet_width": cf_width,
            "cf_subnet_depth": int(len(inference_cfg.hidden_sizes) or 2),
            "cf_dropout": float(inference_cfg.dropout),
            "cf_activation": "silu",
        }
    else:
        raise ValueError(f"Unknown inference network type: {inference_cfg.network_type}")

    composite_space = hpo.CompositeSearchSpace(
        inference_space=inference_space,
        summary_space=summary_space,
    )
    params = {**summary_params, **inference_params}
    summary_net = build_summary_network(params=params, search_space=composite_space)
    inference_net = build_inference_network(params=params, search_space=composite_space)
    return summary_net, inference_net


def _default_ancova_workflow() -> WorkflowConfig:
    """Default WorkflowConfig for ANCOVA: FlowMatching inference network."""
    return WorkflowConfig(
        inference_network=InferenceNetworkConfig(
            network_type="FlowMatching",
            widths=(125, 125, 125),
            dropout=0.05,
            use_optimal_transport=True,
        )
    )


@dataclass
class ANCOVAConfig:
    """
    Complete configuration bundle for ANCOVA 2-arms model.

    This wraps the generic WorkflowConfig with ANCOVA-specific configurations.
    The default inference network is FlowMatching; use
    ``InferenceNetworkConfig(network_type="CouplingFlow")`` to switch back.
    """

    prior: PriorConfig = field(default_factory=PriorConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    workflow: WorkflowConfig = field(default_factory=_default_ancova_workflow)

    def to_dict(self) -> dict:
        """Serialize all configs to nested dict for JSON storage."""
        return {
            "prior": asdict(self.prior),
            "meta": asdict(self.meta),
            "workflow": self.workflow.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ANCOVAConfig":
        """Reconstruct from dict."""
        return cls(
            prior=PriorConfig(**d.get("prior", {})),
            meta=MetaConfig(**d.get("meta", {})),
            workflow=WorkflowConfig.from_dict(d.get("workflow", {})),
        )


# =============================================================================
# ANCOVA-Specific Simulator Functions
# =============================================================================


def prior(
    prior_df: float, prior_scale: float, config: PriorConfig, rng: np.random.Generator
) -> dict:
    """
    Sample parameters for model: outcome = b_covariate*x + b_group*group + noise.

    Parameters
    ----------
    prior_df : float
        Degrees of freedom for b_group prior. If df <= 0 or > 100, uses Normal.
    prior_scale : float
        Scale parameter for b_group prior distribution.
    config : PriorConfig
        Prior configuration with b_covariate_scale.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    dict with b_covariate and b_group arrays (shape (1,))

    Notes
    -----
    - b_covariate ~ Normal(0, config.b_covariate_scale)
    - b_group ~ t(prior_df, 0, prior_scale) or Normal if df <= 0 or df > 100
    - sigma is fixed at 1.0 (not a parameter)
    """
    b_covariate = rng.normal(loc=0, scale=config.b_covariate_scale, size=1).astype(
        np.float64
    )

    b_group = np.array(
        [
            sample_t_or_normal(
                df=float(np.asarray(prior_df).flat[0]),
                scale=float(np.asarray(prior_scale).flat[0]),
                rng=rng,
            )
        ],
        dtype=np.float64,
    )

    return dict(b_covariate=b_covariate, b_group=b_group)


def likelihood(
    b_covariate: float,
    b_group: float,
    n_total: int,
    p_alloc: float,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate 2-arm ANCOVA data with fixed sigma = 1.

    Model: outcome = b_covariate * covariate + b_group * group + noise

    Parameters
    ----------
    b_covariate : float
        Coefficient for baseline covariate.
    b_group : float
        Treatment effect (group difference).
    n_total : int
        Total sample size.
    p_alloc : float
        Probability of treatment allocation (0 to 1).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    dict with outcome, covariate, group arrays (each shape (N,))
    """
    b_cov = float(np.asarray(b_covariate).reshape(-1)[0])
    b_grp = float(np.asarray(b_group).reshape(-1)[0])
    sigma = 1.0  # Fixed
    n_total = int(np.asarray(n_total).reshape(-1)[0])
    p = float(np.clip(p_alloc, 0.01, 0.99))

    # Ensure both groups represented
    max_tries = 1000
    for _ in range(max_tries):
        group = rng.choice([0, 1], size=n_total, p=[1 - p, p])
        if np.sum(group == 0) > 0 and np.sum(group == 1) > 0:
            break
    else:
        # Fallback: force at least 1 in each group
        n_treat = max(1, int(n_total * p))
        n_ctrl = n_total - n_treat
        group = np.concatenate([np.zeros(n_ctrl), np.ones(n_treat)])
        rng.shuffle(group)

    covariate = rng.normal(0, 1, size=n_total)
    y_mean = b_cov * covariate + b_grp * group
    outcome = rng.normal(y_mean, sigma, size=n_total)

    return dict(outcome=outcome, covariate=covariate, group=group)


def meta(config: MetaConfig, rng: np.random.Generator) -> dict:
    """
    Sample meta parameters (context) including prior hyperparameters.

    Parameters
    ----------
    config : MetaConfig
        Configuration with sampling ranges.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    dict with N, p_alloc, prior_df, prior_scale
    """
    n_total = loguniform_int(config.n_min, config.n_max, rng=rng)
    p_alloc = rng.uniform(config.p_alloc_min, config.p_alloc_max)

    # prior_df: log-uniform shifted to allow 0 (Normal)
    prior_df = int(
        round(
            loguniform_int(
                1, config.prior_df_max + 1, alpha=config.prior_df_alpha, rng=rng
            )
            - 1
        )
    )

    prior_scale = rng.gamma(
        shape=config.prior_scale_gamma_shape,
        scale=config.prior_scale_gamma_scale,
    )

    return dict(
        N=n_total,
        p_alloc=p_alloc,
        prior_df=prior_df,
        prior_scale=prior_scale,
    )


def simulate_cond_batch(
    n_sims: int,
    n_total: int,
    p_alloc: float,
    b_covariate: float,
    b_group: float,
    prior_df: float,
    prior_scale: float,
    rng: np.random.Generator = None,
) -> dict:
    """
    Vectorized batch simulation for a single condition.

    Parameters
    ----------
    n_sims : int
        Number of simulations to run.
    n_total : int
        Sample size per simulation.
    p_alloc : float
        Treatment allocation probability.
    b_covariate : float
        Coefficient for baseline covariate.
    b_group : float
        Treatment effect (true value).
    prior_df : float
        Degrees of freedom for prior (context for inference).
    prior_scale : float
        Scale for prior (context for inference).
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.

    Returns
    -------
    dict with outcome, covariate, group matrices (n_sims x n_total) and metadata
    """
    if rng is None:
        rng = np.random.default_rng()

    n_sims = int(n_sims)
    n_total = int(n_total)
    p = float(np.clip(p_alloc, 0.01, 0.99))
    b_cov = float(b_covariate)
    b_grp = float(b_group)

    group = (rng.random((n_sims, n_total)) < p).astype(np.float64)
    covariate = rng.standard_normal((n_sims, n_total))
    noise = rng.standard_normal((n_sims, n_total))
    outcome = b_cov * covariate + b_grp * group + noise

    return {
        "outcome": outcome,
        "covariate": covariate,
        "group": group,
        "N": n_total,
        "p_alloc": p_alloc,
        "prior_df": prior_df,
        "prior_scale": prior_scale,
    }


# =============================================================================
# ANCOVA-Specific Adapter Specification
# =============================================================================


def get_ancova_adapter_spec() -> AdapterSpec:
    """
    Return the adapter specification for ANCOVA 2-arms model.

    This declaratively defines how ANCOVA data should be processed
    by the BayesFlow adapter. The spec includes:
    - Set-based data: outcome, covariate, group (per-observation)
    - Parameters to infer: b_group (treatment effect)
    - Context variables: N, p_alloc, prior_df, prior_scale
    - Standardization of outcome, covariate, and b_group
    - Broadcasting and transformations for context variables

    Returns
    -------
    AdapterSpec
        Declarative specification for ANCOVA adapter.
    """
    return AdapterSpec(
        set_keys=["outcome", "covariate", "group"],
        param_keys=["b_group"],
        context_keys=["N", "p_alloc", "prior_df", "prior_scale"],
        standardize_keys=["outcome", "covariate"],
        prior_standardize={"b_group": (None, "prior_scale")},
        broadcast_specs={
            "N": "outcome",
            "p_alloc": "outcome",
            "prior_df": "outcome",
            "prior_scale": "outcome",
        },
        context_transforms={
            "N": (np.sqrt, np.square),
            "prior_df": (np.log1p, np.expm1),
        },
        output_dtype="float32",
    )


# =============================================================================
# ANCOVA-Specific Factory Functions
# =============================================================================


def create_ancova_workflow_components(config: ANCOVAConfig) -> tuple:
    """
    Create all ANCOVA workflow components using infrastructure.

    This is the main factory function for creating ANCOVA-specific
    summary network, inference network, and adapter.

    Parameters
    ----------
    config : ANCOVAConfig
        Complete ANCOVA configuration.

    Returns
    -------
    tuple
        (summary_net, inference_net, adapter)

    Examples
    --------
    >>> config = ANCOVAConfig()
    >>> summary_net, inference_net, adapter = create_ancova_workflow_components(config)
    """
    summary_net, inference_net = _build_networks_from_workflow_config(config.workflow)
    adapter = create_adapter(get_ancova_adapter_spec())
    return summary_net, inference_net, adapter


def create_prior_fn(config: ANCOVAConfig, rng: np.random.Generator) -> Callable:
    """Create prior function with injected config and rng."""

    def _prior(prior_df, prior_scale):
        return prior(prior_df, prior_scale, config.prior, rng)

    return _prior


def create_likelihood_fn(rng: np.random.Generator) -> Callable:
    """Create likelihood function with injected rng."""

    def _likelihood(b_covariate, b_group, n_total, p_alloc):
        return likelihood(b_covariate, b_group, n_total, p_alloc, rng)

    return _likelihood


def create_meta_fn(config: ANCOVAConfig, rng: np.random.Generator) -> Callable:
    """Create meta function with injected config and rng."""

    def _meta():
        return meta(config.meta, rng)

    return _meta


def create_simulator(
    config: ANCOVAConfig, rng: np.random.Generator = None
) -> bf.simulators.Simulator:
    """
    Create BayesFlow simulator for ANCOVA model.

    Parameters
    ----------
    config : ANCOVAConfig
        Configuration bundle.
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.

    Returns
    -------
    bf.simulators.Simulator configured for ANCOVA 2-arms
    """
    if rng is None:
        rng = np.random.default_rng()

    prior_fn = create_prior_fn(config, rng)
    likelihood_fn = create_likelihood_fn(rng)
    meta_fn = create_meta_fn(config, rng)

    return create_generic_simulator(prior_fn, likelihood_fn, meta_fn)


def create_ancova_adapter() -> "Adapter":
    """
    Create adapter for ANCOVA 2-arms model.

    This is a convenience wrapper that creates an adapter using the
    ANCOVA adapter specification. It's equivalent to:

        from bayesflow_hpo.builders import create_adapter
        adapter = create_adapter(get_ancova_adapter_spec())

    Returns
    -------
    Adapter configured for ANCOVA 2-arms model

    Examples
    --------
    >>> adapter = create_ancova_adapter()
    >>> processed = adapter(simulator.sample(100))
    """
    return create_adapter(get_ancova_adapter_spec())


def create_ancova_objective(
    config: ANCOVAConfig,
    simulator: "Simulator",
    adapter: "Adapter",
    search_space: Any,
    validation_conditions: list[dict],
    n_sims: int = 500,
    n_post_draws: int = 500,
    rng: np.random.Generator = None,
) -> Callable:
    """
    Create Optuna objective function for ANCOVA model optimization.

    This is a lightweight wrapper around `bayesflow_hpo.GenericObjective`
    that provides ANCOVA-specific parameter keys and inference conditions.

    Parameters
    ----------
    config : ANCOVAConfig
        ANCOVA configuration with training settings
    simulator : Simulator
        BayesFlow simulator for the ANCOVA model
    adapter : Adapter
        BayesFlow adapter for data transformation
    search_space : bayesflow_hpo.CompositeSearchSpace
        Search space for hyperparameter sampling
    validation_conditions : list of dict
        Conditions grid for validation (from create_validation_grid)
    n_sims : int, default=500
        Number of simulations per condition for validation
    n_post_draws : int, default=500
        Number of posterior draws per simulation
    rng : np.random.Generator, optional
        Random number generator for reproducibility

    Returns
    -------
    objective : Callable[[Trial], Tuple[float, float]]
        Objective function that takes an Optuna trial and returns
        (calibration_error, parameter_score)

    Examples
    --------
    >>> config = ANCOVAConfig()
    >>> simulator = create_simulator(config, RNG)
    >>> adapter = create_ancova_adapter()
    >>> search_space = hpo.CompositeSearchSpace(...)
    >>> conditions = create_validation_grid(extended=False)
    >>>
    >>> objective = create_ancova_objective(
    ...     config, simulator, adapter, search_space, conditions
    ... )
    >>> study.optimize(objective, n_trials=30)
    """
    if not isinstance(search_space, hpo.CompositeSearchSpace):
        raise TypeError(
            "search_space must be a bayesflow_hpo.CompositeSearchSpace instance."
        )

    validation_data = _build_validation_dataset_from_conditions(
        validation_conditions=validation_conditions,
        n_sims=n_sims,
        rng=rng,
    )

    objective_config = hpo.ObjectiveConfig(
        simulator=simulator,
        adapter=adapter,
        search_space=search_space,
        inference_conditions=["N", "p_alloc", "prior_df", "prior_scale"],
        validation_data=validation_data,
        epochs=int(config.workflow.training.epochs),
        batches_per_epoch=int(config.workflow.training.batches_per_epoch),
        early_stopping_patience=int(config.workflow.training.early_stopping_patience),
        early_stopping_window=int(config.workflow.training.early_stopping_window),
        n_posterior_samples=int(n_post_draws),
    )
    return hpo.GenericObjective(objective_config)


# =============================================================================
# Training Helpers for train_until_threshold (ANCOVA-Specific)
# =============================================================================


def create_ancova_training_functions(
    simulator: "Simulator",
    adapter: "Adapter",
    validation_conditions: list[dict],
    rng: np.random.Generator,
) -> tuple[Callable, Callable, Callable]:
    """
    Create workflow builder, trainer, and validator functions for ANCOVA model.

    These functions are designed to work with `train_until_threshold` from
    the optimization module. Returns ANCOVA-specific implementations with
    the correct parameter keys and inference conditions.

    Parameters
    ----------
    simulator : Simulator
        BayesFlow simulator for the ANCOVA model
    adapter : Adapter
        BayesFlow adapter for data transformation
    validation_conditions : list of dict
        Conditions grid for validation (typically from
        create_validation_grid(extended=True))
    rng : np.random.Generator
        Random number generator for reproducibility

    Returns
    -------
    tuple of (build_workflow_fn, train_fn, validate_fn)
        - build_workflow_fn(params) -> workflow
        - train_fn(workflow) -> history
        - validate_fn(workflow) -> metrics

    Examples
    --------
    >>> config = ANCOVAConfig()
    >>> simulator = create_simulator(config, RNG)
    >>> adapter = create_ancova_adapter()
    >>> conditions = create_validation_grid(extended=True)
    >>>
    >>> build_fn, train_fn, validate_fn = create_ancova_training_functions(
    ...     simulator, adapter, conditions, RNG
    ... )
    >>>
    >>> result = train_until_threshold(
    ...     build_workflow_fn=build_fn,
    ...     train_fn=train_fn,
    ...     validate_fn=validate_fn,
    ...     hyperparams=best_params,
    ...     thresholds=QualityThresholds(),
    ... )
    """
    validation_data = _build_validation_dataset_from_conditions(
        validation_conditions=validation_conditions,
        n_sims=1000,
        rng=rng,
    )

    def build_workflow_fn(params):
        """Build a fresh workflow from hyperparameters."""
        workflow_config = params_dict_to_workflow_config(params)
        summary_net, inference_net = _build_networks_from_workflow_config(
            workflow_config
        )

        return build_workflow(
            simulator=simulator,
            adapter=adapter,
            inference_network=inference_net,
            summary_network=summary_net,
            params={
                "batch_size": int(params["batch_size"]),
                "initial_lr": float(params["initial_lr"]),
                "decay_rate": float(params.get("decay_rate", 0.85)),
            },
            config=WorkflowBuildConfig(
                inference_conditions=["N", "p_alloc", "prior_df", "prior_scale"],
                checkpoint_name="ancova_threshold_workflow",
            ),
        )

    def train_fn(workflow):
        """Train the workflow."""
        # Compile the approximator (required before training)
        try:
            workflow.approximator.compile(optimizer=workflow.optimizer)
        except Exception:
            pass  # Already compiled

        early_stop = hpo.MovingAverageEarlyStopping(
            window=10, patience=10, restore_best_weights=True
        )
        return workflow.fit_online(
            epochs=50,
            batch_size=320,
            num_batches_per_epoch=50,
            validation_data=1000,
            callbacks=[early_stop],
        )

    def validate_fn(workflow):
        """Validate the workflow on the strict grid."""
        results = hpo.run_validation_pipeline(
            approximator=workflow.approximator,
            validation_data=validation_data,
            n_posterior_samples=1000,
        )
        return results["metrics"]

    return build_workflow_fn, train_fn, validate_fn


# =============================================================================
# Validation Helpers (ANCOVA-Specific)
# =============================================================================


def create_validation_grid(extended: bool = False) -> list[dict]:
    """
    Generate conditions for systematic validation.

    Parameters
    ----------
    extended : bool
        If True, include more conditions for comprehensive validation.

    Returns
    -------
    list of condition dicts
    """
    from itertools import product

    if extended:
        # Extended grid for final validation
        conditions = []
        for idx, (n, pdf, psc, b_cov, b_grp, p_alloc) in enumerate(
            product(
                [20, 200, 1000],  # N extremes
                [0, 2],  # prior_df: Normal vs low-df t
                [0.1, 5.0],  # prior_scale extremes
                [-1.0, 1.0],  # b_covariate
                [0.0, 0.3, 1.0],  # b_group: null, small, large
                [0.5, 0.9],  # p_alloc
            )
        ):
            conditions.append(
                {
                    "id_cond": idx,
                    "n_total": n,
                    "p_alloc": p_alloc,
                    "b_covariate": b_cov,
                    "b_arm_treat": b_grp,
                    "prior_df": pdf,
                    "prior_scale": psc,
                }
            )
    else:
        # Reduced grid for optimization (faster)
        conditions = []
        for idx, (n, pdf, psc, b_grp) in enumerate(
            product(
                [20, 500],  # N extremes
                [0, 3],  # prior_df: Normal vs moderate t
                [0.5, 5.0],  # prior_scale extremes
                [0.0, 0.5],  # b_group: null vs moderate
            )
        ):
            conditions.append(
                {
                    "id_cond": idx,
                    "n_total": n,
                    "p_alloc": 0.5,
                    "b_covariate": 0.0,
                    "b_arm_treat": b_grp,
                    "prior_df": pdf,
                    "prior_scale": psc,
                }
            )

    return conditions


def make_simulate_fn(
    rng: np.random.Generator = None,
    param_mapping: dict = None,
) -> Callable:
    """
    Create simulation function for validation pipeline.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator.
    param_mapping : dict, optional
        Mapping from condition keys to simulate_cond_batch params.

    Returns
    -------
    callable: simulate_fn(condition, n_sims) -> dict
    """
    if param_mapping is None:
        param_mapping = {
            "n_total": "n_total",
            "p_alloc": "p_alloc",
            "b_covariate": "b_covariate",
            "b_arm_treat": "b_group",
            "prior_df": "prior_df",
            "prior_scale": "prior_scale",
        }

    def simulate_fn(condition: dict, n_sims: int) -> dict:
        kwargs = {"n_sims": n_sims, "rng": rng}
        for cond_key, fn_param in param_mapping.items():
            val = condition.get(cond_key, condition.get(fn_param, 0.0))
            if isinstance(val, np.ndarray):
                val = float(val.flat[0]) if val.size == 1 else val
            kwargs[fn_param] = val
        return simulate_cond_batch(**kwargs)

    return simulate_fn


def make_infer_fn(approximator) -> Callable:
    """
    Create inference function for validation pipeline.

    Parameters
    ----------
    approximator : bf.approximators.Approximator
        Trained BayesFlow approximator.

    Returns
    -------
    callable: infer_fn(data, n_samples) -> np.ndarray
    """
    return _make_bayesflow_infer_fn(
        approximator=approximator,
        param_keys=["b_group"],
        data_keys=["outcome", "covariate", "group"],
    )


def make_condition_infer_fn(
    approximator,
    param_key: str = "b_group",
    data_keys: list[str] | None = None,
) -> Callable:
    """Create an inference callable for ANCOVA condition-grid validation."""
    if data_keys is None:
        data_keys = ["outcome", "covariate", "group"]

    return _make_bayesflow_infer_fn(
        approximator=approximator,
        param_keys=[param_key],
        data_keys=data_keys,
    )


def run_condition_grid_validation(
    conditions_list: list[dict],
    n_sims: int,
    n_post_draws: int,
    simulate_fn: Callable,
    infer_fn: Callable,
    true_param_key: str,
    coverage_levels: list[float] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run ANCOVA condition-grid validation with simulation/inference callables."""
    if coverage_levels is None:
        coverage_levels = [0.5, 0.8, 0.9, 0.95, 0.99]

    timing = {"simulation": 0.0, "inference": 0.0, "metrics": 0.0}
    all_metrics: list[pd.DataFrame] = []
    sim_counter = 0

    for cond_id, condition in enumerate(conditions_list):
        t0 = time.time()
        sim_data = simulate_fn(condition, n_sims)
        timing["simulation"] += time.time() - t0

        t1 = time.time()
        draws = infer_fn(sim_data, n_post_draws)
        timing["inference"] += time.time() - t1

        if draws.ndim == 3 and draws.shape[-1] == 1:
            draws = np.squeeze(draws, axis=-1)

        true_values = np.asarray(sim_data[true_param_key]).reshape(-1)

        t2 = time.time()
        batch_metrics = compute_batch_metrics(
            draws=np.asarray(draws),
            true_values=true_values,
            cond_id=cond_id,
            sim_id_start=sim_counter,
            coverage_levels=coverage_levels,
        )
        timing["metrics"] += time.time() - t2

        all_metrics.append(batch_metrics)
        sim_counter += int(true_values.shape[0])

        if verbose:
            print(f"Condition {cond_id + 1}/{len(conditions_list)} done")

    sim_metrics = pd.concat(all_metrics, ignore_index=True)
    metrics = aggregate_metrics(
        sim_metrics=sim_metrics,
        coverage_levels=coverage_levels,
        n_posterior_samples=n_post_draws,
    )

    return {
        "sim_metrics": sim_metrics,
        "metrics": metrics,
        "timing": timing,
    }


def _build_validation_dataset_from_conditions(
    validation_conditions: list[dict],
    n_sims: int,
    rng: np.random.Generator | None,
) -> hpo.ValidationDataset:
    """Build fixed validation dataset for ANCOVA condition grids."""
    simulate_fn = make_simulate_fn(rng=rng)
    simulations = [simulate_fn(condition, n_sims=n_sims) for condition in validation_conditions]
    return hpo.ValidationDataset(
        simulations=simulations,
        condition_labels=validation_conditions,
        param_keys=["b_group"],
        data_keys=["outcome", "covariate", "group"],
        seed=42,
    )


# =============================================================================
# ANCOVA-Specific Metadata Utilities
# =============================================================================


def get_model_metadata(
    config: ANCOVAConfig,
    validation_results: dict = None,
    extra: dict = None,
) -> dict:
    """
    Collect all ANCOVA-specific reproducibility metadata.

    Parameters
    ----------
    config : ANCOVAConfig
        Configuration used for training.
    validation_results : dict, optional
        Validation metrics to include.
    extra : dict, optional
        Additional metadata to include.

    Returns
    -------
    dict with complete metadata
    """
    return get_workflow_metadata(
        config=config.workflow.to_dict(),
        model_type="ancova_cont_2arms",
        validation_results=validation_results,
        extra={
            "prior_config": asdict(config.prior),
            "meta_config": asdict(config.meta),
            **(extra or {}),
        },
    )


# Alias for backwards compatibility
save_model_with_metadata = save_workflow_with_metadata
load_model_with_metadata = load_workflow_with_metadata
