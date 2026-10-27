"""Recovery and SBC ECDF plots per sample size for a trained ANCOVA model.

Usage::

    KERAS_BACKEND=torch python scripts/plot_diagnostics.py --model runs/v2

Datasets draw design and prior hyperparameters from ``meta`` (as in
training), with N fixed per panel, and d from that dataset's prior, so the
ECDF check is valid marginally over the hyperparameters at each N.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import bayesflow as bf
import keras
import matplotlib

matplotlib.use("Agg")
import numpy as np
from train_baseline import load_config, load_trained

from bayesflow_rct.models.ancova.config import ANCOVAConfig
from bayesflow_rct.models.ancova.simulator import meta
from bayesflow_rct.models.ancova.training import create_ancova_workflow
from bayesflow_rct.models.ancova.validation import simulate_condition


def simulate_at_n(n_total, n_sims, config, rng):
    rows = [
        simulate_condition({**meta(config, rng), "N": n_total}, 1, config, rng)
        for _ in range(n_sims)
    ]
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("runs/v2"))
    parser.add_argument(
        "--n-grid", type=int, nargs="+", default=[20, 30, 50, 200, 1000]
    )
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--posterior-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    out = args.model / "plots"
    out.mkdir(parents=True, exist_ok=True)

    saved = args.model / "config.json"
    config = load_config(saved) if saved.exists() else ANCOVAConfig()
    if (args.model / "approximator.keras").exists():
        approximator = keras.saving.load_model(args.model / "approximator.keras")
    else:  # weights-only save: rebuild from the saved config
        workflow = create_ancova_workflow(config)
        load_trained(workflow, args.model)
        approximator = workflow.approximator
    rng = np.random.default_rng(args.seed)
    for n_total in args.n_grid:
        sims = simulate_at_n(n_total, args.sims, config, rng)
        truth = {"b_group": sims.pop("b_group")}
        sims.pop("b_covariate")
        draws = approximator.sample(conditions=sims, num_samples=args.posterior_samples)
        fig = bf.diagnostics.plots.recovery(draws, truth, variable_names=["d"])
        fig.axes[0].set_title(f"d, N = {n_total}")
        fig.savefig(out / f"recovery_N{n_total}.png", dpi=120, bbox_inches="tight")
        fig = bf.diagnostics.plots.calibration_ecdf(
            draws, truth, variable_names=["d"], difference=True
        )
        fig.axes[0].set_title(f"d, N = {n_total}")
        fig.savefig(out / f"ecdf_N{n_total}.png", dpi=120, bbox_inches="tight")
        print(f"N={n_total} done", flush=True)


if __name__ == "__main__":
    main()
