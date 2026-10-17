"""
ANCOVA Simulator Functions.

Prior, likelihood, meta-parameter sampling, and factory functions for the
2-arm ANCOVA model specified in ``docs/ancova_prior_spec.md``.

Data are simulated on the natural scale (residual SD = 1), so the target
``b_group`` is Cohen's d. Standardization to mean 0 / SD 1 happens in the
adapter, identically for simulated and real data.
"""

from __future__ import annotations

from collections.abc import Callable

import bayesflow as bf
import numpy as np
from scipy import stats

from bayesflow_rct.core.utils import loguniform_int
from bayesflow_rct.models.ancova.config import ANCOVAConfig


def sample_truncated_t(
    df: float,
    scale: float,
    bound: float,
    rng: np.random.Generator,
    size: int | None = None,
) -> np.ndarray:
    """
    Sample ``scale * t(df)`` truncated to ``[-bound, bound]`` by inverse CDF.

    Exact and rejection-free; computed in float64 so the CDF bounds stay
    distinguishable for very heavy tails or wide scales.

    Parameters
    ----------
    df : float
        Degrees of freedom of the Student-t (``>= 1`` per the prior spec).
    scale : float
        Scale of the Student-t.
    bound : float
        Symmetric truncation bound on the scaled draw.
    rng : np.random.Generator
        Random number generator.
    size : int, optional
        Number of draws. ``None`` returns a 0-d array.

    Returns
    -------
    np.ndarray
        Draws in ``[-bound, bound]``.
    """
    z = float(bound) / float(scale)
    lo = stats.t.cdf(-z, df)
    hi = stats.t.cdf(z, df)
    u = rng.uniform(lo, hi, size=size)
    return np.clip(float(scale) * stats.t.ppf(u, df), -bound, bound)


def prior(
    prior_df: float,
    prior_scale: float,
    config: ANCOVAConfig,
    rng: np.random.Generator,
) -> dict:
    """
    Sample the treatment effect and the covariate nuisance parameter.

    ``b_group ~ prior_scale * t(prior_df)`` truncated to ``|d| <= d_bound``;
    the covariate's share of non-treatment variance ``R2 ~ Beta(a, b)`` is
    mapped to ``b_covariate = ±sqrt(R2 / (1 - R2))``.

    Parameters
    ----------
    prior_df, prior_scale : float
        Hyperparameters of the ``b_group`` prior (sampled in :func:`meta`).
    config : ANCOVAConfig
        Configuration with ``d_bound`` and the R² Beta parameters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    dict with ``b_group`` and ``b_covariate`` arrays of shape (1,).
    """
    df = float(np.asarray(prior_df).flat[0])
    scale = float(np.asarray(prior_scale).flat[0])
    b_group = sample_truncated_t(df, scale, config.d_bound, rng, size=1)

    r2 = rng.beta(config.r2_beta_a, config.r2_beta_b)
    sign = rng.choice([-1.0, 1.0])
    b_covariate = np.array([sign * np.sqrt(r2 / (1.0 - r2))])

    return dict(b_group=b_group, b_covariate=b_covariate)


def allocate(n_total: int, p_alloc: float, rng: np.random.Generator) -> np.ndarray:
    """
    Fixed allocation: exactly ``round(N * p_alloc)`` treated, in random order.

    At least one subject is kept in each arm.
    """
    n_treat = int(np.clip(round(n_total * p_alloc), 1, n_total - 1))
    group = np.zeros(n_total)
    group[:n_treat] = 1.0
    rng.shuffle(group)
    return group


def likelihood(
    b_covariate: float,
    b_group: float,
    n_total: int,
    p_alloc: float,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate raw 2-arm ANCOVA data with residual SD = 1.

    Model: ``outcome = b_group * group + b_covariate * covariate + N(0, 1)``.

    Returns
    -------
    dict with ``outcome``, ``covariate``, ``group`` arrays of shape (N,).
    """
    b_cov = float(np.asarray(b_covariate).flat[0])
    b_grp = float(np.asarray(b_group).flat[0])
    n_total = int(np.asarray(n_total).flat[0])

    group = allocate(n_total, float(p_alloc), rng)
    covariate = rng.standard_normal(n_total)
    outcome = b_grp * group + b_cov * covariate + rng.standard_normal(n_total)
    return dict(outcome=outcome, covariate=covariate, group=group)


def meta(config: ANCOVAConfig, rng: np.random.Generator) -> dict:
    """
    Sample design conditions and prior hyperparameters (one per batch).

    Returns
    -------
    dict with ``N``, ``p_alloc``, ``prior_df``, ``prior_scale``.
    """
    return dict(
        N=int(round(loguniform_int(config.n_min, config.n_max, rng=rng))),
        p_alloc=rng.uniform(config.p_alloc_min, config.p_alloc_max),
        prior_df=1.0
        + rng.lognormal(np.log(config.prior_df_median_shifted), config.prior_df_sdlog),
        prior_scale=rng.lognormal(
            np.log(config.prior_scale_median), config.prior_scale_sdlog
        ),
    )


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

    def _likelihood(b_covariate, b_group, p_alloc, N):  # noqa: N803 (simulator key)
        return likelihood(b_covariate, b_group, N, p_alloc, rng)

    return _likelihood


def create_meta_fn(config: ANCOVAConfig, rng: np.random.Generator) -> Callable:
    """Create meta function with injected config and rng."""

    def _meta():
        return meta(config, rng)

    return _meta


def create_simulator(
    config: ANCOVAConfig, rng: np.random.Generator | None = None
) -> bf.simulators.Simulator:
    """
    Create the BayesFlow simulator for the ANCOVA model.

    Meta conditions (N, p_alloc, prior hyperparameters) are drawn once per
    batch, so all datasets in a batch share the same N.

    Parameters
    ----------
    config : ANCOVAConfig
        Configuration bundle.
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.

    Returns
    -------
    bf.simulators.Simulator
    """
    if rng is None:
        rng = np.random.default_rng()

    return bf.simulators.make_simulator(
        [create_prior_fn(config, rng), create_likelihood_fn(rng)],
        meta_fn=create_meta_fn(config, rng),
    )
