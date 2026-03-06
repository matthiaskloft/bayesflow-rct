"""ANCOVA-specific thin wrapper around bayesflow_hpo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bayesflow_hpo as hpo

from bayesflow_rct.models.ancova.model import (
    ANCOVAConfig,
    create_ancova_adapter,
    create_simulator,
)

VALIDATION_DATA_DIR = Path("data") / "ancova_validation"


def get_or_create_validation_data(
    simulator: Any,
    seed: int = 42,
    path: str | Path = VALIDATION_DATA_DIR,
) -> hpo.ValidationDataset:
    """Load cached ANCOVA validation data or generate it once."""
    dataset_path = Path(path)
    try:
        return hpo.load_validation_dataset(dataset_path)
    except FileNotFoundError:
        val_data = hpo.generate_validation_dataset(
            simulator=simulator,
            param_keys=["b_group"],
            data_keys=["outcome", "covariate", "group"],
            condition_grid={
                "N": [30, 100, 300],
                "p_alloc": [0.5, 0.7],
                "prior_df": [3, 10],
                "prior_scale": [0.5, 1.0],
            },
            sims_per_condition=200,
            seed=seed,
        )
        hpo.save_validation_dataset(val_data, dataset_path)
        return val_data


def run_ancova_hpo(
    n_trials: int = 100,
    storage: str = "sqlite:///ancova_hpo.db",
    study_name: str = "ancova_hpo",
    seed: int = 42,
    search_space: hpo.CompositeSearchSpace | None = None,
):
    """Run Optuna HPO for ANCOVA using bayesflow_hpo."""
    config = ANCOVAConfig()
    simulator = create_simulator(config)
    adapter = create_ancova_adapter()
    val_data = get_or_create_validation_data(simulator=simulator, seed=seed)

    if search_space is None:
        search_space = hpo.CompositeSearchSpace(
            inference_space=hpo.CouplingFlowSpace(
                depth=hpo.IntDimension("cf_depth", 2, 8),
                subnet_width=hpo.IntDimension("cf_subnet_width", 32, 128, step=16),
            ),
            summary_space=hpo.DeepSetSpace(
                summary_dim=hpo.IntDimension("ds_summary_dim", 4, 16),
                width=hpo.IntDimension("ds_width", 32, 128, step=16),
            ),
            training_space=hpo.TrainingSpace(),
        )

    return hpo.optimize(
        simulator=simulator,
        adapter=adapter,
        param_keys=["b_group"],
        data_keys=["outcome", "covariate", "group"],
        validation_data=val_data,
        search_space=search_space,
        inference_conditions=["N", "p_alloc", "prior_df", "prior_scale"],
        n_trials=n_trials,
        storage=storage,
        study_name=study_name,
    )
