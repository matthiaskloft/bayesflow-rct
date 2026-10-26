"""Tests for the ANCOVA adapter."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from bayesflow_rct.models.ancova.adapter import (
    DatasetStandardize,
    SufficientStatistics,
    create_ancova_adapter,
)
from bayesflow_rct.models.ancova.config import ANCOVAConfig
from bayesflow_rct.models.ancova.simulator import create_simulator
from bayesflow_rct.models.ancova.validation import simulate_condition


def _batch(n=6):
    return create_simulator(ANCOVAConfig(), rng=np.random.default_rng(0)).sample(n)


def test_dataset_standardize_per_dataset():
    x = np.random.default_rng(0).normal(3.0, 5.0, size=(4, 50))
    z = DatasetStandardize().forward(x)
    np.testing.assert_allclose(z.mean(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=-1, ddof=1), 1.0, atol=1e-12)


def test_canonical_keys_and_shapes():
    batch = _batch()
    out = create_ancova_adapter()(batch)
    n = batch["outcome"].shape[1]
    assert set(out) == {
        "inference_variables",
        "summary_variables",
        "inference_conditions",
    }
    assert out["inference_variables"].shape == (6, 1)
    assert out["summary_variables"].shape == (6, n, 3)
    assert out["inference_conditions"].shape == (6, 9)
    assert all(v.dtype == np.float32 for v in out.values())


def test_outcome_and_covariate_standardized():
    s = create_ancova_adapter()(_batch())["summary_variables"]
    for k in (0, 1):
        np.testing.assert_allclose(s[..., k].mean(axis=1), 0.0, atol=1e-5)
        np.testing.assert_allclose(s[..., k].std(axis=1, ddof=1), 1.0, atol=1e-5)


def test_invariant_to_affine_rescaling_of_raw_data():
    """Shifting/scaling the raw outcome must not change the network input."""
    batch = _batch()
    shifted = {**batch, "outcome": 7.0 + 3.0 * batch["outcome"]}
    adapter = create_ancova_adapter()
    np.testing.assert_allclose(
        adapter(batch)["summary_variables"],
        adapter(shifted)["summary_variables"],
        atol=1e-5,
    )


def test_validation_batches_match_training_layout():
    cond = {"N": 30, "p_alloc": 0.5, "prior_df": 3.0, "prior_scale": 0.7}
    batch = simulate_condition(cond, 5, ANCOVAConfig(), np.random.default_rng(0))
    out = create_ancova_adapter()(batch)
    assert out["summary_variables"].shape == (5, 30, 3)
    assert out["inference_conditions"].shape == (5, 9)


def test_constant_covariate_raises():
    import pytest

    batch = _batch()
    batch["covariate"] = np.zeros_like(batch["covariate"])
    with pytest.raises(ValueError, match="zero"):
        create_ancova_adapter()(batch)


def test_integer_group_input_becomes_float32():
    batch = _batch()
    batch["group"] = batch["group"].astype(np.int64)
    out = create_ancova_adapter()(batch)
    assert out["summary_variables"].dtype == np.float32


def test_sufficient_statistics_match_raw_ols():
    """d_hat equals the raw-data OLS group effect over the residual SD."""
    cond = {"N": 60, "p_alloc": 0.5, "prior_df": 3.0, "prior_scale": 0.7}
    batch = simulate_condition(cond, 3, ANCOVAConfig(), np.random.default_rng(1))
    stats = SufficientStatistics()(
        {
            **batch,
            "outcome": DatasetStandardize().forward(batch["outcome"]),
            "covariate": DatasetStandardize().forward(batch["covariate"]),
        }
    )
    for i in range(3):
        y, x, g = batch["outcome"][i], batch["covariate"][i], batch["group"][i]
        design = np.c_[np.ones_like(y), g, x]
        beta, rss, *_ = np.linalg.lstsq(design, y, rcond=None)
        d_ols = beta[1] / np.sqrt(rss[0] / (len(y) - 3))
        np.testing.assert_allclose(stats["d_hat"][i, 0], d_ols, rtol=1e-8)
        r = np.corrcoef(x, y)[0, 1]
        np.testing.assert_allclose(stats["r_xy"][i, 0], r, rtol=1e-8)


def test_d_hat_standard_error_is_calibrated():
    cond = {"N": 50, "p_alloc": 0.5, "prior_df": 30.0, "prior_scale": 0.7}
    batch = simulate_condition(cond, 2000, ANCOVAConfig(), np.random.default_rng(2))
    cond_out = create_ancova_adapter()(batch)["inference_conditions"]
    z = (cond_out[:, 7] - batch["b_group"][:, 0]) / np.exp(cond_out[:, 8])
    assert abs(z.mean()) < 0.1
    assert abs(z.std() - 1.0) < 0.06
