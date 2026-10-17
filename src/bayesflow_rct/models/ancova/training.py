"""
ANCOVA Training Helpers.

Workflow construction and training for a fixed configuration, plus the
Optuna objective and threshold-training callables built on ``bayesflow-hpo``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import bayesflow as bf
import bayesflow_hpo as hpo
import keras
import numpy as np

from bayesflow_rct.models.ancova.adapter import create_ancova_adapter
from bayesflow_rct.models.ancova.config import (
    ANCOVAConfig,
    build_networks,
    hpo_params_to_config,
)
from bayesflow_rct.models.ancova.simulator import create_simulator
from bayesflow_rct.models.ancova.validation import (
    build_validation_dataset,
    run_validation,
)

if TYPE_CHECKING:
    from bayesflow import Adapter, Simulator

OBJECTIVE_METRICS = ("calibration_error", "nrmse")
COST_METRIC = "inference_time"


def create_ancova_workflow_components(config: ANCOVAConfig) -> tuple:
    """
    Create all ANCOVA workflow components.

    Returns
    -------
    tuple of (summary_net, inference_net, adapter)
    """
    summary_net, inference_net = build_networks(config)
    return summary_net, inference_net, create_ancova_adapter()


def make_optimizer(config: ANCOVAConfig) -> keras.optimizers.Optimizer:
    """
    Adam with linear warmup and cosine decay from ``initial_lr`` to ``min_lr``.

    Steps are optimizer steps (batches): the horizon is
    ``epochs * batches_per_epoch``. See Keras ``CosineDecay``
    (https://keras.io/api/optimizers/learning_rate_schedules/cosine_decay/).
    """
    total_steps = int(config.epochs) * int(config.batches_per_epoch)
    warmup = int(config.lr_warmup_steps)
    if warmup >= total_steps:
        raise ValueError(
            f"lr_warmup_steps ({warmup}) must be < total steps ({total_steps})."
        )
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=total_steps - warmup,
        alpha=float(config.min_lr) / float(config.initial_lr),
        warmup_target=float(config.initial_lr),
        warmup_steps=warmup,
    )
    return keras.optimizers.Adam(learning_rate=schedule)


def create_ancova_workflow(
    config: ANCOVAConfig,
    simulator: Simulator | None = None,
    adapter: Adapter | None = None,
    rng: np.random.Generator | None = None,
) -> bf.BasicWorkflow:
    """
    Build a :class:`bayesflow.BasicWorkflow` for a fixed configuration.

    Inference variables and conditions are standardized by the approximator.
    The approximator is compiled here because BasicWorkflow only compiles
    when it builds its own optimizer.
    """
    summary_net, inference_net = build_networks(config)
    workflow = bf.BasicWorkflow(
        simulator=simulator or create_simulator(config, rng),
        adapter=adapter or create_ancova_adapter(),
        inference_network=inference_net,
        summary_network=summary_net,
        optimizer=make_optimizer(config),
        standardize=["inference_variables", "inference_conditions"],
    )
    workflow.approximator.compile(optimizer=workflow.optimizer)
    return workflow


def train_ancova_workflow(
    workflow: bf.BasicWorkflow,
    config: ANCOVAConfig,
    callbacks: list | None = None,
) -> keras.callbacks.History:
    """
    Train online with moving-average early stopping on the validation loss.
    """
    early_stop = hpo.MovingAverageEarlyStopping(
        monitor="val_loss",
        window=int(config.early_stopping_window),
        patience=int(config.early_stopping_patience),
        restore_best_weights=True,
    )
    return workflow.fit_online(
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        num_batches_per_epoch=int(config.batches_per_epoch),
        validation_data=int(config.validation_sims),
        callbacks=[early_stop, *(callbacks or [])],
    )


def create_ancova_objective(
    config: ANCOVAConfig,
    search_space: hpo.CompositeSearchSpace,
    simulator: Simulator | None = None,
    adapter: Adapter | None = None,
    validation_data: hpo.ValidationDataset | None = None,
    n_posterior_samples: int = 500,
) -> hpo.GenericObjective:
    """
    Create the Optuna objective for ANCOVA HPO.

    Uses the ANCOVA validation grid (which ``optimize()`` cannot generate,
    because the grid fixes prior hyperparameters rather than simulator
    keywords) and ``bayesflow-hpo``'s fixed-budget training. The objective
    returns one value per metric in :data:`OBJECTIVE_METRICS` plus the cost
    metric, all minimized.

    Note: ``GenericObjective`` builds its own approximator and optimizer from
    the search space, so ``initial_lr``/``min_lr``/``lr_warmup_steps`` and
    :func:`make_optimizer` apply only to fixed-config training
    (:func:`create_ancova_workflow`), not to HPO trials.
    """
    if not isinstance(search_space, hpo.CompositeSearchSpace):
        raise TypeError(
            "search_space must be a bayesflow_hpo.CompositeSearchSpace instance."
        )
    objective_config = hpo.ObjectiveConfig(
        simulator=simulator or create_simulator(config),
        adapter=adapter or create_ancova_adapter(),
        search_space=search_space,
        validation_data=validation_data or build_validation_dataset(config),
        epochs=int(config.epochs),
        num_batches=int(config.batches_per_epoch),
        n_posterior_samples=int(n_posterior_samples),
        objective_metrics=list(OBJECTIVE_METRICS),
        cost_metric=COST_METRIC,
    )
    return hpo.GenericObjective(objective_config)


def create_ancova_training_functions(
    config: ANCOVAConfig,
    simulator: Simulator,
    adapter: Adapter,
    validation_data: hpo.ValidationDataset,
    n_posterior_samples: int = 1000,
) -> tuple[Callable, Callable, Callable]:
    """
    Create builder, trainer and validator callables for ``train_until_threshold``.

    Returns
    -------
    tuple of (build_workflow_fn, train_fn, validate_fn)
    """
    current: dict[str, ANCOVAConfig] = {"config": config}

    def build_workflow_fn(params: dict[str, Any]) -> bf.BasicWorkflow:
        cfg = hpo_params_to_config(params) if params else config
        current["config"] = cfg
        return create_ancova_workflow(cfg, simulator=simulator, adapter=adapter)

    def train_fn(workflow: bf.BasicWorkflow) -> keras.callbacks.History:
        return train_ancova_workflow(workflow, current["config"])

    def validate_fn(workflow: bf.BasicWorkflow) -> dict[str, Any]:
        result = run_validation(
            workflow.approximator, validation_data, n_posterior_samples
        )
        return result.summary

    return build_workflow_fn, train_fn, validate_fn
