"""
ANCOVA Validation Helpers.

Condition-grid generation, simulation/inference callables, and
the validation loop for evaluating ANCOVA model quality.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from itertools import product
from typing import Any

import bayesflow_hpo as hpo
import numpy as np
from bayesflow_hpo.validation import DEFAULT_METRICS, resolve_metrics
from bayesflow_hpo.validation.inference import (
    make_bayesflow_infer_fn as _make_bayesflow_infer_fn,
)
from bayesflow_hpo.validation.metrics import (
    aggregate_condition_rows,
    compute_condition_metrics,
)

from bayesflow_rct.models.ancova.simulator import simulate_cond_batch

# ---------------------------------------------------------------------------
# Validation grids
# ---------------------------------------------------------------------------


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
    if extended:
        ranges = product(
            [20, 200, 1000],  # N extremes
            [0, 2],  # prior_df: Normal vs low-df t
            [0.1, 5.0],  # prior_scale extremes
            [-1.0, 1.0],  # b_covariate
            [0.0, 0.3, 1.0],  # b_group: null, small, large
            [0.5, 0.9],  # p_alloc
        )
        keys = (
            "n_total", "prior_df", "prior_scale",
            "b_covariate", "b_arm_treat", "p_alloc",
        )
    else:
        ranges = product(
            [20, 500],  # N extremes
            [0, 3],  # prior_df: Normal vs moderate t
            [0.5, 5.0],  # prior_scale extremes
            [0.0, 0.5],  # b_group: null vs moderate
        )
        keys = ("n_total", "prior_df", "prior_scale", "b_arm_treat")

    conditions = []
    for idx, values in enumerate(ranges):
        cond = {"id_cond": idx}
        cond.update(zip(keys, values))
        # Fill fixed defaults for reduced grid
        cond.setdefault("p_alloc", 0.5)
        cond.setdefault("b_covariate", 0.0)
        conditions.append(cond)

    return conditions


# ---------------------------------------------------------------------------
# Simulation / inference callables for validation
# ---------------------------------------------------------------------------

# ANCOVA key mapping: condition dict keys → simulate_cond_batch parameter names
_ANCOVA_KEY_MAPPING = {
    "n_total": "n_total",
    "p_alloc": "p_alloc",
    "b_covariate": "b_covariate",
    "b_arm_treat": "b_group",
    "prior_df": "prior_df",
    "prior_scale": "prior_scale",
}


def make_simulate_fn(
    rng: np.random.Generator = None,
) -> Callable:
    """
    Create simulation function for validation pipeline.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    callable: simulate_fn(condition, n_sims) -> dict
    """

    def simulate_fn(condition: dict, n_sims: int) -> dict:
        kwargs: dict[str, Any] = {"n_sims": n_sims, "rng": rng}
        missing = [
            cond_key
            for cond_key, fn_param in _ANCOVA_KEY_MAPPING.items()
            if cond_key not in condition and fn_param not in condition
        ]
        if missing:
            raise KeyError(
                f"Missing required condition keys: {missing}. "
                f"Expected keys (or aliases): "
                f"{list(_ANCOVA_KEY_MAPPING.items())}"
            )
        for cond_key, fn_param in _ANCOVA_KEY_MAPPING.items():
            val = condition[cond_key] if cond_key in condition else condition[fn_param]
            if isinstance(val, np.ndarray):
                val = float(val.flat[0]) if val.size == 1 else val
            kwargs[fn_param] = val
        return simulate_cond_batch(**kwargs)

    return simulate_fn


def make_condition_infer_fn(
    approximator,
) -> Callable:
    """
    Create an inference callable for ANCOVA condition-grid validation.

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


# Backward-compatible alias
make_infer_fn = make_condition_infer_fn


# ---------------------------------------------------------------------------
# Condition-grid validation loop
# ---------------------------------------------------------------------------


def run_condition_grid_validation(
    conditions_list: list[dict],
    n_sims: int,
    n_post_draws: int,
    simulate_fn: Callable,
    infer_fn: Callable,
    true_param_key: str,
    metric_names: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run ANCOVA condition-grid validation with simulation/inference callables.

    Parameters
    ----------
    conditions_list : list of dict
        Condition dictionaries with parameter values.
    n_sims : int
        Number of simulations per condition.
    n_post_draws : int
        Number of posterior draws per simulation.
    simulate_fn : callable
        Function(condition, n_sims) -> dict of simulated data.
    infer_fn : callable
        Function(data, n_post_draws) -> np.ndarray of posterior draws.
    true_param_key : str
        Key for the true parameter values in simulated data.
    metric_names : list of str, optional
        Metric names to compute. Default: bayesflow_hpo DEFAULT_METRICS.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict with keys 'condition_rows', 'summary', 'timing'
    """
    if metric_names is None:
        metric_names = DEFAULT_METRICS
    metric_fns = resolve_metrics(metric_names)

    timing = {"simulation": 0.0, "inference": 0.0, "metrics": 0.0}
    condition_rows: list[dict[str, Any]] = []

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
        row = compute_condition_metrics(
            draws=np.asarray(draws),
            true_values=true_values,
            cond_id=cond_id,
            metric_fns=metric_fns,
        )
        timing["metrics"] += time.time() - t2

        condition_rows.append(row)

        if verbose:
            print(f"Condition {cond_id + 1}/{len(conditions_list)} done")

    summary = aggregate_condition_rows(condition_rows)

    return {
        "condition_rows": condition_rows,
        "summary": summary,
        "timing": timing,
    }


def build_validation_dataset(
    validation_conditions: list[dict],
    n_sims: int,
    rng: np.random.Generator | None,
) -> hpo.ValidationDataset:
    """Build fixed validation dataset for ANCOVA condition grids."""
    simulate_fn = make_simulate_fn(rng=rng)
    simulations = [
        simulate_fn(condition, n_sims=n_sims) for condition in validation_conditions
    ]
    return hpo.ValidationDataset(
        simulations=simulations,
        condition_labels=validation_conditions,
        param_keys=["b_group"],
        data_keys=["outcome", "covariate", "group"],
        seed=42,
    )
