"""Tests for ANCOVA condition-grid validation helpers."""

import os

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

from bayesflow_rct.models.ancova import validation as ancova_validation


def test_make_condition_infer_fn_uses_defaults(monkeypatch):
	captured: dict[str, object] = {}

	def fake_make_bayesflow_infer_fn(*, approximator, param_keys, data_keys):
		captured["approximator"] = approximator
		captured["param_keys"] = param_keys
		captured["data_keys"] = data_keys

		def _infer(_data, _n_samples):
			return np.zeros((1, 1), dtype=float)

		return _infer

	monkeypatch.setattr(
		ancova_validation,
		"_make_bayesflow_infer_fn",
		fake_make_bayesflow_infer_fn,
	)

	approximator = object()
	infer_fn = ancova_validation.make_condition_infer_fn(approximator=approximator)

	assert callable(infer_fn)
	assert captured["approximator"] is approximator
	assert captured["param_keys"] == ["b_group"]
	assert captured["data_keys"] == ["outcome", "covariate", "group"]


def test_run_condition_grid_validation_returns_expected_structure(monkeypatch):
	"""Test that run_condition_grid_validation returns correct structure."""
	conditions = [{"b_arm_treat": 0.25}]

	def simulate_fn(_condition, n_sims):
		return {
			"b_arm_treat": np.full(n_sims, 0.25, dtype=float),
		}

	def infer_fn(sim_data, n_post_draws):
		n_sims = int(np.asarray(sim_data["b_arm_treat"]).shape[0])
		draws = np.full((n_sims, n_post_draws, 1), 0.25, dtype=float)
		return draws

	# Mock the metric functions to avoid bayesflow dependency issues
	fake_metrics = {
		"mean_cal_error": lambda draws, true: {"mean_cal_error": 0.01},
	}
	monkeypatch.setattr(ancova_validation, "resolve_metrics", lambda _names: fake_metrics)

	# Mock aggregate to return a simple summary
	monkeypatch.setattr(
		ancova_validation,
		"aggregate_condition_rows",
		lambda rows: {"mean_cal_error": 0.01},
	)

	result = ancova_validation.run_condition_grid_validation(
		conditions_list=conditions,
		n_sims=8,
		n_post_draws=10,
		simulate_fn=simulate_fn,
		infer_fn=infer_fn,
		true_param_key="b_arm_treat",
		verbose=False,
	)

	assert set(result.keys()) == {"condition_rows", "summary", "timing"}
	assert len(result["condition_rows"]) == 1
	assert "mean_cal_error" in result["summary"]
