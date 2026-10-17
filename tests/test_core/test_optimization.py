"""Tests for ANCOVA HPO wrapper and validation-data utilities."""

import importlib
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from bayesflow_rct.models.ancova.config import ANCOVAConfig
from bayesflow_rct.models.ancova.validation import build_validation_dataset

ancova_hpo = importlib.import_module("bayesflow_rct.models.ancova.hpo")

SMALL_GRID = {"N": [20, 40], "p_alloc": [0.5], "prior_df": [3.0], "prior_scale": [0.7]}


def test_build_validation_dataset_grid():
    data = build_validation_dataset(condition_grid=SMALL_GRID, sims_per_condition=3)
    assert len(data.simulations) == 2
    assert [c["N"] for c in data.condition_labels] == [20, 40]
    assert data.simulations[1]["outcome"].shape == (3, 40)
    assert data.param_keys == ["b_group"]


def test_build_validation_dataset_is_reproducible():
    a = build_validation_dataset(condition_grid=SMALL_GRID, sims_per_condition=3)
    b = build_validation_dataset(condition_grid=SMALL_GRID, sims_per_condition=3)
    np.testing.assert_array_equal(
        a.simulations[0]["b_group"], b.simulations[0]["b_group"]
    )


def test_get_or_create_validation_data_loads_cached(monkeypatch, tmp_path: Path):
    cached = object()
    monkeypatch.setattr(ancova_hpo.hpo, "load_validation_dataset", lambda p: cached)
    assert ancova_hpo.get_or_create_validation_data(path=tmp_path) is cached


def test_get_or_create_validation_data_generates_and_saves(monkeypatch, tmp_path):
    def fake_load(_path):
        raise FileNotFoundError

    saved = {}
    monkeypatch.setattr(ancova_hpo.hpo, "load_validation_dataset", fake_load)
    monkeypatch.setattr(
        ancova_hpo.hpo,
        "save_validation_dataset",
        lambda dataset, path: saved.update(dataset=dataset, path=path),
    )
    monkeypatch.setattr(
        ancova_hpo,
        "build_validation_dataset",
        lambda config, sims_per_condition, seed: ("data", seed),
    )
    result = ancova_hpo.get_or_create_validation_data(
        ANCOVAConfig(), seed=7, path=tmp_path
    )
    assert result == ("data", 7)
    assert saved == {"dataset": ("data", 7), "path": tmp_path}


def test_run_ancova_hpo_builds_objective_and_study(monkeypatch):
    calls = {}

    class FakeStudy:
        def optimize(self, objective, n_trials):
            calls["optimize"] = (objective, n_trials)

    monkeypatch.setattr(
        ancova_hpo, "get_or_create_validation_data", lambda config, seed: "val"
    )
    monkeypatch.setattr(
        ancova_hpo,
        "create_ancova_objective",
        lambda config, search_space, validation_data: ("obj", validation_data),
    )
    monkeypatch.setattr(
        ancova_hpo.hpo,
        "create_study",
        lambda study_name, storage, directions, metric_names: calls.setdefault(
            "study", (study_name, storage, directions, metric_names)
        )
        and FakeStudy(),
    )
    ancova_hpo.run_ancova_hpo(
        n_trials=3, storage="sqlite:///x.db", study_name="s", search_space="space"
    )
    assert calls["study"] == (
        "s",
        "sqlite:///x.db",
        ["minimize"] * 3,
        ["calibration_error", "nrmse", "inference_time"],
    )
    assert calls["optimize"] == (("obj", "val"), 3)
