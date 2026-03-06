"""Tests for hpo_params_to_config and ANCOVAConfig HPO round-trip."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import pytest

from bayesflow_rct.models.ancova.config import ANCOVAConfig, hpo_params_to_config


class TestHpoParamsToConfigCouplingFlow:
    """Test mapping of CouplingFlow HPO params to ANCOVAConfig."""

    COUPLING_FLOW_PARAMS = {
        "ds_summary_dim": 8,
        "ds_depth": 3,
        "ds_width": 48,
        "ds_dropout": 0.1,
        "cf_depth": 6,
        "cf_subnet_width": 80,
        "cf_subnet_depth": 2,
        "cf_dropout": 0.15,
        "initial_lr": 1e-3,
        "batch_size": 256,
        "decay_rate": 0.9,
    }

    def test_returns_ancova_config(self):
        config = hpo_params_to_config(self.COUPLING_FLOW_PARAMS)
        assert isinstance(config, ANCOVAConfig)

    def test_summary_network_params(self):
        config = hpo_params_to_config(self.COUPLING_FLOW_PARAMS)
        assert config.summary_dim == 8
        assert config.summary_depth == 3
        assert config.summary_width == 48
        assert config.summary_dropout == 0.1

    def test_inference_network_type(self):
        config = hpo_params_to_config(self.COUPLING_FLOW_PARAMS)
        assert config.inference_network_type == "CouplingFlow"

    def test_coupling_flow_architecture(self):
        config = hpo_params_to_config(self.COUPLING_FLOW_PARAMS)
        assert config.inference_depth == 6
        assert config.inference_hidden_sizes == (80, 80)  # width=80, depth=2
        assert config.inference_dropout == 0.15

    def test_training_params(self):
        config = hpo_params_to_config(self.COUPLING_FLOW_PARAMS)
        assert config.initial_lr == 1e-3
        assert config.batch_size == 256
        assert config.decay_rate == 0.9


class TestHpoParamsToConfigFlowMatching:
    """Test mapping of FlowMatching HPO params to ANCOVAConfig."""

    FLOW_MATCHING_PARAMS = {
        "ds_summary_dim": 12,
        "ds_depth": 2,
        "ds_width": 64,
        "ds_dropout": 0.05,
        "fm_subnet_width": 96,
        "fm_subnet_depth": 4,
        "fm_dropout": 0.08,
        "fm_use_ot": True,
        "initial_lr": 5e-4,
        "batch_size": 320,
    }

    def test_inference_network_type(self):
        config = hpo_params_to_config(self.FLOW_MATCHING_PARAMS)
        assert config.inference_network_type == "FlowMatching"

    def test_flow_matching_architecture(self):
        config = hpo_params_to_config(self.FLOW_MATCHING_PARAMS)
        assert config.inference_widths == (96, 96, 96, 96)  # width=96, depth=4
        assert config.inference_dropout == 0.08
        assert config.inference_use_optimal_transport is True

    def test_summary_params_mapped(self):
        config = hpo_params_to_config(self.FLOW_MATCHING_PARAMS)
        assert config.summary_dim == 12
        assert config.summary_depth == 2


class TestHpoParamsToConfigEdgeCases:
    """Test edge cases and defaults."""

    def test_empty_params_returns_defaults(self):
        config = hpo_params_to_config({})
        defaults = ANCOVAConfig()
        assert config.summary_dim == defaults.summary_dim
        assert config.inference_network_type == defaults.inference_network_type

    def test_unknown_keys_ignored(self):
        params = {"unknown_key": 42, "another_unknown": "value"}
        config = hpo_params_to_config(params)
        assert isinstance(config, ANCOVAConfig)

    def test_training_only_params(self):
        """Only training params (no prefix) should work."""
        params = {"initial_lr": 2e-4, "batch_size": 512}
        config = hpo_params_to_config(params)
        assert config.initial_lr == 2e-4
        assert config.batch_size == 512
        # Architecture should be defaults
        defaults = ANCOVAConfig()
        assert config.inference_network_type == defaults.inference_network_type

    def test_facade_re_export(self):
        """hpo_params_to_config should be available via model.py facade."""
        from bayesflow_rct.models.ancova.model import (
            hpo_params_to_config as facade_fn,
        )

        assert facade_fn is hpo_params_to_config


class TestHpoParamsRoundTrip:
    """Test that HPO params → config → build_networks uses correct values."""

    def test_coupling_flow_params_not_lost(self):
        """Architecture params must survive the HPO → config mapping."""
        params = {
            "ds_summary_dim": 16,
            "ds_depth": 4,
            "cf_depth": 8,
            "cf_subnet_width": 64,
            "cf_subnet_depth": 3,
            "batch_size": 256,
        }
        config = hpo_params_to_config(params)

        # Verify architecture params were NOT replaced by defaults
        defaults = ANCOVAConfig()
        assert config.summary_dim == 16, "ds_summary_dim should map to summary_dim"
        assert config.summary_dim != defaults.summary_dim or defaults.summary_dim == 16
        assert config.inference_depth == 8
        assert config.inference_hidden_sizes == (64, 64, 64)
        assert config.batch_size == 256
