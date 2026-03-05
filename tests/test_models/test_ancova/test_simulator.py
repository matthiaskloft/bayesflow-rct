"""Tests for ANCOVA simulator functions."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from bayesflow_rct.models.ancova.config import ANCOVAConfig
from bayesflow_rct.models.ancova.simulator import (
    likelihood,
    meta,
    prior,
    simulate_cond_batch,
)


class TestPrior:
    def test_returns_dict_with_expected_keys(self):
        config = ANCOVAConfig()
        rng = np.random.default_rng(42)
        result = prior(prior_df=3, prior_scale=1.0, config=config, rng=rng)
        assert "b_covariate" in result
        assert "b_group" in result

    def test_output_shapes(self):
        config = ANCOVAConfig()
        rng = np.random.default_rng(42)
        result = prior(prior_df=3, prior_scale=1.0, config=config, rng=rng)
        assert result["b_covariate"].shape == (1,)
        assert result["b_group"].shape == (1,)

    def test_normal_prior_when_df_zero(self):
        """df=0 should use Normal distribution (via sample_t_or_normal)."""
        config = ANCOVAConfig()
        rng = np.random.default_rng(42)
        result = prior(prior_df=0, prior_scale=1.0, config=config, rng=rng)
        assert np.isfinite(result["b_group"][0])


class TestLikelihood:
    def test_returns_dict_with_expected_keys(self):
        rng = np.random.default_rng(42)
        result = likelihood(
            b_covariate=0.5, b_group=0.3, n_total=50, p_alloc=0.5, rng=rng
        )
        assert set(result.keys()) == {"outcome", "covariate", "group"}

    def test_output_shapes(self):
        rng = np.random.default_rng(42)
        result = likelihood(
            b_covariate=0.5, b_group=0.3, n_total=50, p_alloc=0.5, rng=rng
        )
        assert result["outcome"].shape == (50,)
        assert result["covariate"].shape == (50,)
        assert result["group"].shape == (50,)

    def test_both_groups_present(self):
        rng = np.random.default_rng(42)
        result = likelihood(
            b_covariate=0.5, b_group=0.3, n_total=100, p_alloc=0.5, rng=rng
        )
        unique_groups = np.unique(result["group"])
        assert len(unique_groups) == 2


class TestMeta:
    def test_returns_expected_keys(self):
        config = ANCOVAConfig()
        rng = np.random.default_rng(42)
        result = meta(config, rng)
        assert set(result.keys()) == {"N", "p_alloc", "prior_df", "prior_scale"}

    def test_n_in_range(self):
        config = ANCOVAConfig()
        rng = np.random.default_rng(42)
        for _ in range(20):
            result = meta(config, rng)
            assert config.n_min <= result["N"] <= config.n_max

    def test_p_alloc_in_range(self):
        config = ANCOVAConfig()
        rng = np.random.default_rng(42)
        for _ in range(20):
            result = meta(config, rng)
            assert config.p_alloc_min <= result["p_alloc"] <= config.p_alloc_max

    def test_prior_df_in_range(self):
        config = ANCOVAConfig()
        rng = np.random.default_rng(42)
        for _ in range(20):
            result = meta(config, rng)
            assert config.prior_df_min <= result["prior_df"] <= config.prior_df_max


class TestSimulateCondBatch:
    def test_returns_expected_keys(self):
        result = simulate_cond_batch(
            n_sims=10, n_total=50, p_alloc=0.5,
            b_covariate=0.5, b_group=0.3,
            prior_df=3, prior_scale=1.0,
            rng=np.random.default_rng(42),
        )
        expected = {"outcome", "covariate", "group", "N", "p_alloc", "prior_df", "prior_scale"}
        assert set(result.keys()) == expected

    def test_output_shapes(self):
        result = simulate_cond_batch(
            n_sims=10, n_total=50, p_alloc=0.5,
            b_covariate=0.5, b_group=0.3,
            prior_df=3, prior_scale=1.0,
            rng=np.random.default_rng(42),
        )
        assert result["outcome"].shape == (10, 50)
        assert result["covariate"].shape == (10, 50)
        assert result["group"].shape == (10, 50)

    def test_metadata_values(self):
        result = simulate_cond_batch(
            n_sims=10, n_total=50, p_alloc=0.5,
            b_covariate=0.5, b_group=0.3,
            prior_df=3, prior_scale=1.0,
            rng=np.random.default_rng(42),
        )
        assert result["N"] == 50
        assert result["p_alloc"] == 0.5
        assert result["prior_df"] == 3
        assert result["prior_scale"] == 1.0
