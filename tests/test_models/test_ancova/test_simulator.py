"""Tests for ANCOVA simulator functions."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pytest
from scipy import stats

from bayesflow_rct.models.ancova.config import ANCOVAConfig
from bayesflow_rct.models.ancova.simulator import (
    allocate,
    create_simulator,
    likelihood,
    meta,
    prior,
    sample_truncated_t,
)


class TestTruncatedT:
    def test_respects_bound(self):
        rng = np.random.default_rng(0)
        draws = sample_truncated_t(df=1.0, scale=3.0, bound=5.0, rng=rng, size=5000)
        assert np.all(np.abs(draws) <= 5.0)

    def test_matches_truncated_distribution(self):
        """KS test against scipy's t restricted to [-bound, bound]."""
        rng = np.random.default_rng(1)
        df, scale, bound = 3.0, 0.7, 2.0
        draws = sample_truncated_t(df, scale, bound, rng, size=4000)
        lo, hi = stats.t.cdf(-bound / scale, df), stats.t.cdf(bound / scale, df)

        def cdf(x):
            return (stats.t.cdf(x / scale, df) - lo) / (hi - lo)

        assert stats.kstest(draws, cdf).pvalue > 0.01


class TestPrior:
    def test_keys_and_shapes(self):
        result = prior(3.0, 1.0, ANCOVAConfig(), np.random.default_rng(42))
        assert result["b_group"].shape == (1,)
        assert result["b_covariate"].shape == (1,)

    def test_b_group_truncated(self):
        config = ANCOVAConfig()
        rng = np.random.default_rng(3)
        draws = [prior(1.0, 3.0, config, rng)["b_group"][0] for _ in range(500)]
        assert np.max(np.abs(draws)) <= config.d_bound

    def test_covariate_r2_mapping(self):
        """b_cov = ±sqrt(R2/(1-R2)) implies R2 = b²/(1+b²) has the Beta prior."""
        config = ANCOVAConfig()
        rng = np.random.default_rng(4)
        b = np.array(
            [prior(3.0, 1.0, config, rng)["b_covariate"][0] for _ in range(3000)]
        )
        r2 = b**2 / (1 + b**2)
        beta = stats.beta(config.r2_beta_a, config.r2_beta_b)
        assert stats.kstest(r2, beta.cdf).pvalue > 0.01


class TestAllocation:
    @pytest.mark.parametrize("n,p", [(20, 0.5), (101, 0.8), (30, 0.4)])
    def test_exact_counts(self, n, p):
        group = allocate(n, p, np.random.default_rng(0))
        assert group.sum() == round(n * p)

    def test_both_arms_nonempty(self):
        group = allocate(20, 0.99, np.random.default_rng(0))
        assert 0 < group.sum() < 20


class TestLikelihood:
    def test_keys_and_shapes(self):
        result = likelihood(0.5, 0.3, 50, 0.5, np.random.default_rng(42))
        assert set(result) == {"outcome", "covariate", "group"}
        assert all(v.shape == (50,) for v in result.values())

    def test_residual_sd_is_one(self):
        rng = np.random.default_rng(5)
        d = likelihood(1.0, 0.5, 20000, 0.5, rng)
        resid = d["outcome"] - 0.5 * d["group"] - 1.0 * d["covariate"]
        assert abs(resid.std() - 1.0) < 0.03


class TestMeta:
    def test_ranges(self):
        config = ANCOVAConfig()
        rng = np.random.default_rng(42)
        for _ in range(200):
            m = meta(config, rng)
            assert set(m) == {"N", "p_alloc", "prior_df", "prior_scale"}
            assert config.n_min <= m["N"] <= config.n_max
            assert isinstance(m["N"], int)
            assert config.p_alloc_min <= m["p_alloc"] <= config.p_alloc_max
            assert m["prior_df"] >= 1.0
            assert m["prior_scale"] > 0.0


def test_simulator_batch_shapes():
    simulator = create_simulator(ANCOVAConfig(), rng=np.random.default_rng(0))
    batch = simulator.sample(4)
    n = batch["outcome"].shape[1]
    assert batch["b_group"].shape == (4, 1)
    assert batch["covariate"].shape == (4, n)
    assert batch["group"].shape == (4, n)


@pytest.mark.parametrize(
    ("share", "alpha", "small", "large"),
    [(1.0, 1.0, 0.100, 0.177), (0.7, 0.7, 0.143, 0.243)],
)
def test_sample_n_shares(share, alpha, small, large):
    """Shares at N < 30 and N >= 500: pure log-uniform default and v3 mixture."""
    from bayesflow_rct.models.ancova.simulator import sample_n

    config = ANCOVAConfig(n_log_share=share, n_log_alpha=alpha)
    rng = np.random.default_rng(0)
    n = np.array([sample_n(config, rng) for _ in range(20000)])
    assert n.min() >= config.n_min and n.max() <= config.n_max
    assert abs((n < 30).mean() - small) < 0.01
    assert abs((n >= 500).mean() - large) < 0.01


class TestSimulateBatch:
    COND = {"N": 40, "p_alloc": 0.7, "prior_df": 3.0, "prior_scale": 0.7}

    def test_matches_looped_prior(self):
        """Vectorized draws have the same b_group and R² distributions."""
        from bayesflow_rct.models.ancova.simulator import simulate_batch

        config = ANCOVAConfig()
        vec = simulate_batch(config, np.random.default_rng(0), 3000, self.COND)
        rng = np.random.default_rng(1)
        loop = [prior(3.0, 0.7, config, rng) for _ in range(3000)]
        b_loop = np.array([p["b_group"][0] for p in loop])
        assert stats.ks_2samp(vec["b_group"][:, 0], b_loop).pvalue > 0.01
        r2 = vec["b_covariate"][:, 0] ** 2 / (1 + vec["b_covariate"][:, 0] ** 2)
        beta = stats.beta(config.r2_beta_a, config.r2_beta_b)
        assert stats.kstest(r2, beta.cdf).pvalue > 0.01

    def test_exact_allocation_and_residual_sd(self):
        from bayesflow_rct.models.ancova.simulator import simulate_batch

        out = simulate_batch(ANCOVAConfig(), np.random.default_rng(2), 500, self.COND)
        assert np.all(out["group"].sum(axis=1) == round(40 * 0.7))
        resid = (
            out["outcome"]
            - out["b_group"] * out["group"]
            - out["b_covariate"] * out["covariate"]
        )
        assert abs(resid.std() - 1.0) < 0.02
        assert out["N"] == 40
