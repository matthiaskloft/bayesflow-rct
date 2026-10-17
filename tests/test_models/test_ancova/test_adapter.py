"""Tests for the ANCOVA adapter."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from bayesflow_rct.models.ancova.adapter import (
    DatasetStandardize,
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
    assert out["inference_conditions"].shape == (6, 4)
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
    assert out["inference_conditions"].shape == (5, 4)


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
