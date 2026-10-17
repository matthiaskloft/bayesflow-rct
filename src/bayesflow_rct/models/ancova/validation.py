"""
ANCOVA Validation Helpers.

Fixed condition-grid validation data for simulation-based calibration (SBC):
for every grid point the design and prior hyperparameters are fixed, and
``b_group`` is drawn from that prior, so rank statistics are valid per
condition.
"""

from __future__ import annotations

import itertools
from typing import Any

import bayesflow_hpo as hpo
import numpy as np

from bayesflow_rct.models.ancova.adapter import CONTEXT_KEYS, DATA_KEYS
from bayesflow_rct.models.ancova.config import ANCOVAConfig
from bayesflow_rct.models.ancova.simulator import likelihood, prior

PARAM_KEYS = ["b_group"]

#: Default grid: smallest/largest N, balanced vs. unbalanced allocation,
#: Cauchy / rctbayespower default t(3) / near-Normal priors, and sceptical /
#: JZS-default / wide prior scales.
DEFAULT_CONDITION_GRID: dict[str, list[Any]] = {
    "N": [20, 50, 200, 1000],
    "p_alloc": [0.5, 0.8],
    "prior_df": [1.0, 3.0, 30.0],
    "prior_scale": [0.3, 0.7, 2.0],
}


def simulate_condition(
    condition: dict[str, Any],
    n_sims: int,
    config: ANCOVAConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Simulate ``n_sims`` datasets for one fixed condition, drawing parameters
    from the prior implied by that condition.

    Parameters
    ----------
    condition : dict
        Values for ``N``, ``p_alloc``, ``prior_df``, ``prior_scale``.
    n_sims : int
        Number of datasets.
    config : ANCOVAConfig
        Configuration (truncation bound, R² prior).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    dict
        Batched arrays: parameters ``(n_sims, 1)``, data ``(n_sims, N)``,
        context ``(n_sims, 1)``.
    """
    n_total = int(condition["N"])
    p_alloc = float(condition["p_alloc"])
    rows = []
    for _ in range(int(n_sims)):
        theta = prior(condition["prior_df"], condition["prior_scale"], config, rng)
        data = likelihood(
            theta["b_covariate"], theta["b_group"], n_total, p_alloc, rng
        )
        rows.append({**theta, **data})

    batch = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    for key in CONTEXT_KEYS:
        batch[key] = np.full((int(n_sims), 1), float(condition[key]))
    return batch


def build_validation_dataset(
    config: ANCOVAConfig | None = None,
    condition_grid: dict[str, list[Any]] | None = None,
    sims_per_condition: int = 200,
    seed: int = 42,
) -> hpo.ValidationDataset:
    """
    Build the fixed ANCOVA validation dataset over a condition grid.

    Parameters
    ----------
    config : ANCOVAConfig, optional
        Configuration; defaults to ``ANCOVAConfig()``.
    condition_grid : dict, optional
        ``{key: [values]}`` over ``N``, ``p_alloc``, ``prior_df``,
        ``prior_scale``; defaults to :data:`DEFAULT_CONDITION_GRID`.
    sims_per_condition : int
        Datasets per grid point.
    seed : int
        Seed for reproducibility.

    Returns
    -------
    bayesflow_hpo.ValidationDataset
    """
    config = config or ANCOVAConfig()
    grid = condition_grid or DEFAULT_CONDITION_GRID
    missing = set(CONTEXT_KEYS) - set(grid)
    if missing:
        raise KeyError(f"condition_grid is missing keys: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    labels = [
        dict(zip(CONTEXT_KEYS, values, strict=True))
        for values in itertools.product(*(grid[k] for k in CONTEXT_KEYS))
    ]
    simulations = [
        simulate_condition(label, sims_per_condition, config, rng)
        for label in labels
    ]
    return hpo.ValidationDataset(
        simulations=simulations,
        condition_labels=labels,
        param_keys=PARAM_KEYS,
        data_keys=DATA_KEYS + CONTEXT_KEYS,
        seed=seed,
    )


def run_validation(
    approximator: Any,
    validation_data: hpo.ValidationDataset,
    n_posterior_samples: int = 1000,
) -> Any:
    """
    Run the bayesflow-hpo validation pipeline on the ANCOVA grid.

    Returns
    -------
    bayesflow_hpo ValidationResult (``.summary``, per-condition rows).
    """
    return hpo.run_validation_pipeline(
        approximator=approximator,
        validation_data=validation_data,
        n_posterior_samples=n_posterior_samples,
    )
