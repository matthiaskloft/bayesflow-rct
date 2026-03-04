"""Tests for slim core infrastructure."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from bayesflow_hpo.builders import PriorStandardize as HpoPriorStandardize

from bayesflow_rct.core.infrastructure import PriorStandardize, create_simulator


def test_prior_standardize_is_hpo_export():
    assert PriorStandardize is HpoPriorStandardize


def test_create_simulator_returns_bayesflow_simulator():
    def prior_fn():
        return {"theta": 0.0}

    def likelihood_fn(theta):
        del theta
        return {"y": 0.0}

    simulator = create_simulator(prior_fn, likelihood_fn)
    assert hasattr(simulator, "sample")
