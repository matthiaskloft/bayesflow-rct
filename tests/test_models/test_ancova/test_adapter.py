"""Tests for ANCOVA adapter specification and construction."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from bayesflow_rct.models.ancova.adapter import (
    create_ancova_adapter,
    get_ancova_adapter_spec,
)


class TestANCOVAAdapterSpec:
    def test_spec_returns_dict(self):
        spec = get_ancova_adapter_spec()
        assert isinstance(spec, dict)

    def test_spec_has_required_keys(self):
        spec = get_ancova_adapter_spec()
        assert "set_keys" in spec
        assert "param_keys" in spec
        assert "context_keys" in spec

    def test_set_keys(self):
        spec = get_ancova_adapter_spec()
        assert spec["set_keys"] == ["outcome", "covariate", "group"]

    def test_param_keys(self):
        spec = get_ancova_adapter_spec()
        assert spec["param_keys"] == ["b_group"]

    def test_context_keys(self):
        spec = get_ancova_adapter_spec()
        assert spec["context_keys"] == ["N", "p_alloc", "prior_df", "prior_scale"]

    def test_prior_standardize_spec(self):
        spec = get_ancova_adapter_spec()
        assert "prior_standardize" in spec
        assert "b_group" in spec["prior_standardize"]

    def test_context_transforms(self):
        spec = get_ancova_adapter_spec()
        transforms = spec["context_transforms"]
        assert "N" in transforms
        assert "prior_df" in transforms
        # Each transform is a (forward, inverse) tuple
        fwd, inv = transforms["N"]
        assert np.isclose(fwd(4.0), 2.0)  # sqrt(4) = 2
        assert np.isclose(inv(2.0), 4.0)  # 2^2 = 4


class TestANCOVAAdapterCreation:
    def test_create_returns_adapter(self):
        adapter = create_ancova_adapter()
        # Should be a BayesFlow Adapter instance
        assert hasattr(adapter, "__call__")

    def test_facade_import(self):
        """Adapter functions should be importable from model facade."""
        from bayesflow_rct.models.ancova.model import (
            create_ancova_adapter as facade_create,
            get_ancova_adapter_spec as facade_spec,
        )

        assert facade_create is create_ancova_adapter
        assert facade_spec is get_ancova_adapter_spec
