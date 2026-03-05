"""Tests for ANCOVA metadata utilities."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from bayesflow_rct.models.ancova.config import ANCOVAConfig
from bayesflow_rct.models.ancova.metadata import get_model_metadata


class TestGetModelMetadata:
    def test_returns_dict(self):
        config = ANCOVAConfig()
        metadata = get_model_metadata(config=config)
        assert isinstance(metadata, dict)

    def test_contains_model_type(self):
        config = ANCOVAConfig()
        metadata = get_model_metadata(config=config)
        assert metadata["model_type"] == "ancova_cont_2arms"

    def test_contains_config(self):
        config = ANCOVAConfig()
        metadata = get_model_metadata(config=config)
        assert "config" in metadata
        assert metadata["config"]["inference_network_type"] == "FlowMatching"

    def test_includes_validation_results(self):
        config = ANCOVAConfig()
        val_results = {"cal_error": 0.05, "converged": True}
        metadata = get_model_metadata(
            config=config, validation_results=val_results
        )
        assert "validation" in metadata
        assert metadata["validation"]["cal_error"] == 0.05

    def test_facade_import(self):
        from bayesflow_rct.models.ancova.model import get_model_metadata as facade_fn

        assert facade_fn is get_model_metadata
