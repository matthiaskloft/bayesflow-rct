"""Tests for ANCOVA HPO wrapper utilities."""

import os
from pathlib import Path
import importlib

os.environ.setdefault("KERAS_BACKEND", "torch")

ancova_hpo = importlib.import_module("bayesflow_rct.models.ancova.hpo")


def test_get_or_create_validation_data_loads_cached(monkeypatch, tmp_path: Path):
	cached_dataset = object()

	def fake_load(path):
		assert path == tmp_path
		return cached_dataset

	monkeypatch.setattr(ancova_hpo.hpo, "load_validation_dataset", fake_load)

	result = ancova_hpo.get_or_create_validation_data(simulator=object(), path=tmp_path)
	assert result is cached_dataset


def test_get_or_create_validation_data_generates_when_missing(
	monkeypatch,
	tmp_path: Path,
):
	generated_dataset = object()
	calls: dict[str, object] = {}

	def fake_load(_path):
		raise FileNotFoundError("not cached")

	def fake_generate(**kwargs):
		calls["generate"] = kwargs
		return generated_dataset

	def fake_save(dataset, path):
		calls["save"] = {"dataset": dataset, "path": path}

	monkeypatch.setattr(ancova_hpo.hpo, "load_validation_dataset", fake_load)
	monkeypatch.setattr(ancova_hpo.hpo, "generate_validation_dataset", fake_generate)
	monkeypatch.setattr(ancova_hpo.hpo, "save_validation_dataset", fake_save)

	simulator = object()
	result = ancova_hpo.get_or_create_validation_data(
		simulator=simulator,
		seed=123,
		path=tmp_path,
	)

	assert result is generated_dataset
	assert calls["generate"]["simulator"] is simulator
	assert calls["generate"]["seed"] == 123
	assert calls["save"]["dataset"] is generated_dataset
	assert calls["save"]["path"] == tmp_path


def test_run_ancova_hpo_delegates_to_optimize(monkeypatch):
	calls: dict[str, object] = {}

	simulator = object()
	adapter = object()
	validation_data = object()
	search_space = object()

	monkeypatch.setattr(ancova_hpo, "create_simulator", lambda _config: simulator)
	monkeypatch.setattr(ancova_hpo, "create_ancova_adapter", lambda: adapter)
	monkeypatch.setattr(
		ancova_hpo,
		"get_or_create_validation_data",
		lambda simulator, seed: validation_data,
	)

	def fake_optimize(**kwargs):
		calls["optimize"] = kwargs
		return {"status": "ok"}

	monkeypatch.setattr(ancova_hpo.hpo, "optimize", fake_optimize)

	result = ancova_hpo.run_ancova_hpo(
		n_trials=7,
		storage="sqlite:///tmp.db",
		study_name="ancova_test",
		seed=99,
		search_space=search_space,
	)

	assert result == {"status": "ok"}
	optimize_kwargs = calls["optimize"]
	assert optimize_kwargs["simulator"] is simulator
	assert optimize_kwargs["adapter"] is adapter
	assert optimize_kwargs["validation_data"] is validation_data
	assert optimize_kwargs["search_space"] is search_space
	assert optimize_kwargs["n_trials"] == 7
	assert optimize_kwargs["storage"] == "sqlite:///tmp.db"
	assert optimize_kwargs["study_name"] == "ancova_test"
