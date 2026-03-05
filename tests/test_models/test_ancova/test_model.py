"""Tests for ANCOVA model config and structure."""

import os
from dataclasses import asdict

os.environ.setdefault("KERAS_BACKEND", "torch")

from bayesflow_rct.models.ancova.config import ANCOVAConfig


class TestANCOVAConfigDefaults:
    def test_default_inference_network_is_flow_matching(self):
        config = ANCOVAConfig()
        assert config.inference_network_type == "FlowMatching"

    def test_default_flow_matching_widths(self):
        config = ANCOVAConfig()
        assert config.inference_widths == (125, 125, 125)

    def test_default_flow_matching_dropout(self):
        config = ANCOVAConfig()
        assert config.inference_dropout == 0.05

    def test_override_network_type(self):
        config = ANCOVAConfig(inference_network_type="CouplingFlow")
        assert config.inference_network_type == "CouplingFlow"

    def test_independent_defaults(self):
        """Each ANCOVAConfig() instance should have independent state."""
        c1 = ANCOVAConfig()
        c2 = ANCOVAConfig()
        c1.inference_dropout = 0.99
        assert c2.inference_dropout != 0.99

    def test_serialization_roundtrip(self):
        config = ANCOVAConfig()
        d = asdict(config)
        restored = ANCOVAConfig(**d)
        assert restored.inference_network_type == "FlowMatching"
        assert restored.inference_widths == (125, 125, 125)

    def test_hpo_params_roundtrip(self):
        """HPO results (flat dict) should map 1:1 to config fields."""
        params = {
            "summary_dim": 16,
            "summary_depth": 4,
            "initial_lr": 1e-3,
            "batch_size": 128,
        }
        config = ANCOVAConfig(**params)
        assert config.summary_dim == 16
        assert config.summary_depth == 4
        assert config.initial_lr == 1e-3
        assert config.batch_size == 128

    def test_facade_import_still_works(self):
        """model.py facade should still export ANCOVAConfig."""
        from bayesflow_rct.models.ancova.model import ANCOVAConfig as FacadeConfig

        assert FacadeConfig is ANCOVAConfig
