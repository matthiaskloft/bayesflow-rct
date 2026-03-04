"""Tests for ANCOVA model."""

import os
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

from bayesflow_rct.models.ancova.model import ANCOVAConfig, InferenceNetworkConfig


class TestANCOVAConfigDefaults:
    def test_default_inference_network_is_flow_matching(self):
        config = ANCOVAConfig()
        assert config.workflow.inference_network.network_type == "FlowMatching"

    def test_default_flow_matching_widths(self):
        config = ANCOVAConfig()
        assert config.workflow.inference_network.widths == (125, 125, 125)

    def test_default_flow_matching_dropout(self):
        config = ANCOVAConfig()
        assert config.workflow.inference_network.dropout == 0.05

    def test_override_to_coupling_flow(self):
        config = ANCOVAConfig()
        config.workflow.inference_network = InferenceNetworkConfig(network_type="CouplingFlow")
        assert config.workflow.inference_network.network_type == "CouplingFlow"

    def test_independent_defaults(self):
        """Each ANCOVAConfig() instance should have its own workflow object."""
        c1 = ANCOVAConfig()
        c2 = ANCOVAConfig()
        c1.workflow.inference_network.dropout = 0.99
        assert c2.workflow.inference_network.dropout != 0.99

    def test_serialization_roundtrip(self):
        config = ANCOVAConfig()
        d = config.to_dict()
        restored = ANCOVAConfig.from_dict(d)
        assert restored.workflow.inference_network.network_type == "FlowMatching"
        assert tuple(restored.workflow.inference_network.widths) == (125, 125, 125)
