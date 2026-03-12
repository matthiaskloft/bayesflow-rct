"""
ANCOVA Simulator Functions.

Prior, likelihood, meta-parameter sampling, and factory functions
for creating BayesFlow simulators.
"""

from __future__ import annotations

from collections.abc import Callable

import bayesflow as bf
import numpy as np

from bayesflow_rct.core.utils import loguniform_int, sample_t_or_normal
from bayesflow_rct.models.ancova.config import ANCOVAConfig


def prior(
    prior_df: float,
    prior_scale: float,
    config: ANCOVAConfig,
    rng: np.random.Generator,
) -> dict:
    """
    Sample parameters for model: outcome = b_covariate*x + b_group*group + noise.

    Parameters
    ----------
    prior_df : float
        Degrees of freedom for b_group prior. If df <= 0 or > 100, uses Normal.
    prior_scale : float
        Scale parameter for b_group prior distribution.
    config : ANCOVAConfig
        Configuration with b_covariate_scale.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    dict with b_covariate and b_group arrays (shape (1,))
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


def meta(config: ANCOVAConfig, rng: np.random.Generator) -> dict:
    """
    Sample meta parameters (context) including prior hyperparameters.

    Parameters
    ----------
    config : ANCOVAConfig
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


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_prior_fn(config: ANCOVAConfig, rng: np.random.Generator) -> Callable:
    """Create prior function with injected config and rng."""

    def _prior(prior_df, prior_scale):
        return prior(prior_df, prior_scale, config, rng)

    return _prior


def create_likelihood_fn(rng: np.random.Generator) -> Callable:
    """Create likelihood function with injected rng."""

    def _likelihood(
        b_covariate,
        b_group,
        p_alloc,
        n_total=None,
        **kwargs,
    ):
        # BayesFlow condition grids commonly provide sample size as `N`.
        if n_total is None:
            n_total = kwargs.get("N")
            if n_total is None:
                raise TypeError("Expected either `n_total` or `N` in likelihood inputs")
        return likelihood(b_covariate, b_group, n_total, p_alloc, rng)

    return _likelihood


def create_meta_fn(config: ANCOVAConfig, rng: np.random.Generator) -> Callable:
    """Create meta function with injected config and rng."""

    def _meta():
        return meta(config, rng)

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

    return bf.simulators.make_simulator([prior_fn, likelihood_fn], meta_fn=meta_fn)
