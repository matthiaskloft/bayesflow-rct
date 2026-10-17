"""ANCOVA-specific thin wrapper around bayesflow_hpo."""

from __future__ import annotations

from pathlib import Path

import bayesflow_hpo as hpo
import optuna

from bayesflow_rct.models.ancova.config import ANCOVAConfig
from bayesflow_rct.models.ancova.training import (
    COST_METRIC,
    OBJECTIVE_METRICS,
    create_ancova_objective,
)
from bayesflow_rct.models.ancova.validation import build_validation_dataset

VALIDATION_DATA_DIR = Path("data") / "ancova_validation"


def get_or_create_validation_data(
    config: ANCOVAConfig | None = None,
    seed: int = 42,
    path: str | Path = VALIDATION_DATA_DIR,
    sims_per_condition: int = 200,
) -> hpo.ValidationDataset:
    """Load cached ANCOVA validation data or generate and cache it once."""
    dataset_path = Path(path)
    try:
        return hpo.load_validation_dataset(dataset_path)
    except FileNotFoundError:
        val_data = build_validation_dataset(
            config, sims_per_condition=sims_per_condition, seed=seed
        )
        hpo.save_validation_dataset(val_data, dataset_path)
        return val_data


def default_search_space() -> hpo.CompositeSearchSpace:
    """FlowMatching + DeepSet search space with the default training space."""
    return hpo.CompositeSearchSpace(
        inference_space=hpo.FlowMatchingSpace(),
        summary_space=hpo.DeepSetSpace(),
        training_space=hpo.TrainingSpace(),
    )


def run_ancova_hpo(
    n_trials: int = 100,
    storage: str = "sqlite:///ancova_hpo.db",
    study_name: str = "ancova_hpo",
    seed: int = 42,
    search_space: hpo.CompositeSearchSpace | None = None,
    config: ANCOVAConfig | None = None,
) -> optuna.Study:
    """Run multi-objective Optuna HPO for ANCOVA using bayesflow_hpo."""
    config = config or ANCOVAConfig()
    objective = create_ancova_objective(
        config,
        search_space=search_space or default_search_space(),
        validation_data=get_or_create_validation_data(config, seed=seed),
    )
    metric_names = [*OBJECTIVE_METRICS, COST_METRIC]
    study = hpo.create_study(
        study_name=study_name,
        storage=storage,
        directions=["minimize"] * len(metric_names),
        metric_names=metric_names,
    )
    study.optimize(objective, n_trials=n_trials)
    return study
