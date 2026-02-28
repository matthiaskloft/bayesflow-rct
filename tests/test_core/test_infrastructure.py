"""Tests for core BayesFlow infrastructure."""

import os
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

import bayesflow as bf

from rctbp_bf_training.core.infrastructure import (
    InferenceNetworkConfig,
    build_inference_network,
    params_dict_to_workflow_config,
)


# =============================================================================
# InferenceNetworkConfig
# =============================================================================

class TestInferenceNetworkConfig:
    def test_default_is_coupling_flow(self):
        config = InferenceNetworkConfig()
        assert config.network_type == "CouplingFlow"

    def test_flow_matching_config(self):
        config = InferenceNetworkConfig(network_type="FlowMatching", widths=(128, 128), dropout=0.05)
        assert config.network_type == "FlowMatching"
        assert config.widths == (128, 128)
        assert config.dropout == 0.05

    def test_coupling_flow_fields_present(self):
        config = InferenceNetworkConfig()
        assert hasattr(config, "depth")
        assert hasattr(config, "hidden_sizes")

    def test_flow_matching_fields_present(self):
        config = InferenceNetworkConfig()
        assert hasattr(config, "widths")
        assert hasattr(config, "use_optimal_transport")


# =============================================================================
# build_inference_network
# =============================================================================

class TestBuildInferenceNetwork:
    def test_coupling_flow(self):
        config = InferenceNetworkConfig(network_type="CouplingFlow", depth=4, hidden_sizes=(64, 64))
        net = build_inference_network(config)
        assert isinstance(net, bf.networks.CouplingFlow)

    def test_flow_matching(self):
        config = InferenceNetworkConfig(network_type="FlowMatching", widths=(64, 64))
        net = build_inference_network(config)
        assert isinstance(net, bf.networks.FlowMatching)

    def test_unknown_type_raises(self):
        config = InferenceNetworkConfig(network_type="MagicFlow")
        with pytest.raises(ValueError, match="Unknown inference network type"):
            build_inference_network(config)

    def test_flow_matching_no_optimal_transport_by_default(self):
        config = InferenceNetworkConfig(network_type="FlowMatching")
        net = build_inference_network(config)
        assert isinstance(net, bf.networks.FlowMatching)


# =============================================================================
# params_dict_to_workflow_config
# =============================================================================

class TestParamsDictToWorkflowConfig:
    def test_defaults_give_coupling_flow(self):
        config = params_dict_to_workflow_config({})
        assert config.inference_network.network_type == "CouplingFlow"

    def test_flow_matching_via_network_type_key(self):
        config = params_dict_to_workflow_config({"network_type": "FlowMatching"})
        assert config.inference_network.network_type == "FlowMatching"

    def test_flow_matching_widths_from_depth_and_width(self):
        config = params_dict_to_workflow_config({
            "network_type": "FlowMatching",
            "flow_depth": 3,
            "flow_width": 128,
        })
        assert config.inference_network.widths == (128, 128, 128)

    def test_coupling_flow_depth_and_hidden(self):
        config = params_dict_to_workflow_config({"flow_depth": 5, "flow_hidden": 64})
        assert config.inference_network.depth == 5
        assert config.inference_network.hidden_sizes == (64, 64)

    def test_summary_network_params_passed_through(self):
        config = params_dict_to_workflow_config({"summary_dim": 16, "deepset_depth": 4})
        assert config.summary_network.summary_dim == 16
        assert config.summary_network.depth == 4
