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
from bayesflow_rct.models.ancova.simulator import create_simulator, meta
from bayesflow_rct.models.ancova.validation import (
    build_validation_dataset,
    run_validation,
    simulate_condition,
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


class MultiSizeValidation(keras.utils.PyDataset):
    """
    Fixed, pre-adapted validation batches spanning ``config.validation_n_grid``.

    Each batch has one N (sets must share a shape) and its own draw of
    ``p_alloc``/``prior_df``/``prior_scale`` from :func:`meta`, mirroring the
    training batches.
    """

    def __init__(self, config: ANCOVAConfig, adapter: Adapter, seed: int = 7):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.batches = []
        for n_total in config.validation_n_grid:
            for _ in range(int(config.validation_batches_per_n)):
                cond = {**meta(config, rng), "N": int(n_total)}
                sims = simulate_condition(
                    cond, int(config.validation_batch_size), config, rng
                )
                self.batches.append(adapter(sims))

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return self.batches[index]


def train_ancova_workflow(
    workflow: bf.BasicWorkflow,
    config: ANCOVAConfig,
    callbacks: list | None = None,
) -> keras.callbacks.History:
    """
    Train online for a fixed budget, validating on :class:`MultiSizeValidation`.

    Bypasses ``BasicWorkflow.fit_online``, which wraps validation data in a
    single-shape ``OfflineDataset``. Moving-average early stopping is added
    only if ``config.use_early_stopping``.
    """
    callbacks = list(callbacks or [])
    if config.use_early_stopping:
        callbacks.insert(
            0,
            hpo.MovingAverageEarlyStopping(
                monitor="val_loss",
                window=int(config.early_stopping_window),
                patience=int(config.early_stopping_patience),
                restore_best_weights=True,
            ),
        )
    dataset = bf.datasets.OnlineDataset(
        simulator=workflow.simulator,
        batch_size=int(config.batch_size),
        num_batches=int(config.batches_per_epoch),
        adapter=workflow.adapter,
    )
    return workflow.approximator.fit(
        dataset=dataset,
        epochs=int(config.epochs),
        validation_data=MultiSizeValidation(config, workflow.adapter),
        callbacks=callbacks,
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
