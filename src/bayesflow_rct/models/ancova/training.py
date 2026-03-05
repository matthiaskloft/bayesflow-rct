"""
ANCOVA Training Helpers.

Workflow construction, Optuna objective, and threshold-based training
functions specific to the ANCOVA model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import bayesflow_hpo as hpo
from bayesflow_hpo.builders import (
    WorkflowBuildConfig,
    build_workflow,
)

from bayesflow_rct.models.ancova.adapter import create_ancova_adapter
from bayesflow_rct.models.ancova.config import ANCOVAConfig, build_networks
from bayesflow_rct.models.ancova.validation import build_validation_dataset

if TYPE_CHECKING:
    import numpy as np
    from bayesflow import Adapter, Simulator


def create_ancova_workflow_components(config: ANCOVAConfig) -> tuple:
    """
    Create all ANCOVA workflow components.

    Parameters
    ----------
    config : ANCOVAConfig
        Complete ANCOVA configuration.

    Returns
    -------
    tuple of (summary_net, inference_net, adapter)
    """
    summary_net, inference_net = build_networks(config)
    adapter = create_ancova_adapter()
    return summary_net, inference_net, adapter


def create_ancova_objective(
    config: ANCOVAConfig,
    simulator: Simulator,
    adapter: Adapter,
    search_space: Any,
    validation_conditions: list[dict],
    n_sims: int = 500,
    n_post_draws: int = 500,
    rng: np.random.Generator = None,
) -> Callable:
    """
    Create Optuna objective function for ANCOVA model optimization.

    Parameters
    ----------
    config : ANCOVAConfig
        ANCOVA configuration with training settings.
    simulator : Simulator
        BayesFlow simulator for the ANCOVA model.
    adapter : Adapter
        BayesFlow adapter for data transformation.
    search_space : bayesflow_hpo.CompositeSearchSpace
        Search space for hyperparameter sampling.
    validation_conditions : list of dict
        Conditions grid for validation.
    n_sims : int, default=500
        Number of simulations per condition for validation.
    n_post_draws : int, default=500
        Number of posterior draws per simulation.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    objective : Callable
        Objective function for Optuna.
    """

    if not isinstance(search_space, hpo.CompositeSearchSpace):
        raise TypeError(
            "search_space must be a bayesflow_hpo.CompositeSearchSpace instance."
        )

    validation_data = build_validation_dataset(
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
        epochs=int(config.epochs),
        batches_per_epoch=int(config.batches_per_epoch),
        early_stopping_patience=int(config.early_stopping_patience),
        early_stopping_window=int(config.early_stopping_window),
        n_posterior_samples=int(n_post_draws),
    )
    return hpo.GenericObjective(objective_config)


def create_ancova_training_functions(
    simulator: Simulator,
    adapter: Adapter,
    validation_conditions: list[dict],
    rng: np.random.Generator,
) -> tuple[Callable, Callable, Callable]:
    """
    Create workflow builder, trainer, and validator functions for ANCOVA model.

    These functions are designed to work with ``train_until_threshold`` from
    the threshold module.

    Parameters
    ----------
    simulator : Simulator
        BayesFlow simulator for the ANCOVA model.
    adapter : Adapter
        BayesFlow adapter for data transformation.
    validation_conditions : list of dict
        Conditions grid for validation.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    tuple of (build_workflow_fn, train_fn, validate_fn)
    """

    validation_data = build_validation_dataset(
        validation_conditions=validation_conditions,
        n_sims=1000,
        rng=rng,
    )

    def build_workflow_fn(params):
        """Build a fresh workflow from hyperparameters."""
        config = ANCOVAConfig(**{
            k: v for k, v in params.items()
            if k in ANCOVAConfig.__dataclass_fields__
        })
        summary_net, inference_net = build_networks(config)

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
