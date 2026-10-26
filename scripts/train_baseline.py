"""Train and validate the fixed-config ANCOVA baseline (Cohen's d) model.

Usage::

    KERAS_BACKEND=torch python scripts/train_baseline.py --out runs/v2
    # re-validate saved weights with another solver, no training:
    KERAS_BACKEND=torch python scripts/train_baseline.py --validate-only \
        --out runs/baseline --integrate-steps 16

Writes the trained approximator, training history, SBC summary and the
tail-calibration table for P(d > c) to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd
from bayesflow_hpo.validation import (
    DEFAULT_METRICS,
    aggregate_condition_rows,
    compute_condition_metrics,
    resolve_metrics,
)

from bayesflow_rct.models.ancova.config import ANCOVAConfig
from bayesflow_rct.models.ancova.training import (
    create_ancova_workflow,
    train_ancova_workflow,
)
from bayesflow_rct.models.ancova.validation import (
    DEFAULT_CONDITION_GRID,
    build_validation_dataset,
)

THRESHOLDS = (-0.5, -0.2, 0.0, 0.2, 0.5)
PROB_BINS = np.array([0.0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 1.0])
#: One-sided lower credible bounds q_alpha; P(d > q_alpha) should be 1 - alpha.
ALPHAS = (0.025, 0.05, 0.1, 0.2)
#: Efficacy rules "claim d > c if P(d > c) >= gamma"; futility rules
#: "stop if P(d > c) <= 1 - gamma".
GAMMAS = (0.8, 0.9, 0.95, 0.975)
#: Datasets per sampling call.
CHUNK = 100


def validate(approximator, val_data, n_samples: int):
    """Sample each condition once; return per-condition metrics and tables.

    One-sided focus, both directions:

    - ``lower_cov_<1-alpha>`` (efficacy side): share of datasets whose true d
      exceeds the posterior alpha-quantile; ``upper_cov_<1-alpha>`` (futility
      side): share below the (1 - alpha)-quantile. Nominal ``1 - alpha``.
    - decision table: efficacy rules P(d > c) >= gamma and futility rules
      P(d > c) <= 1 - gamma, with the number of decisions, the mean predicted
      P(d > c) and the observed share with true d > c. Calibrated if the
      observed share is close to the mean prediction; for futility a higher
      observed share means stopping truly effective treatments.
    - reliability table: binned predicted P(d > c) vs observed frequency.

    Two-sided SBC metrics from bayesflow-hpo are kept for reference.
    """
    metric_fns = resolve_metrics(list(DEFAULT_METRICS))
    metric_rows, rows = [], []
    for cond_id, (label, sims) in enumerate(
        zip(val_data.condition_labels, val_data.simulations, strict=True)
    ):
        conditions = {k: v for k, v in sims.items() if k != "b_group"}
        # Chunk over datasets: one call over 500 x 1000 draws exhausts 6 GB.
        n_sims = len(sims["b_group"])
        draws = np.concatenate(
            [
                np.asarray(
                    approximator.sample(
                        conditions={k: v[i : i + CHUNK] for k, v in conditions.items()},
                        num_samples=n_samples,
                    )["b_group"]
                )[..., 0]
                for i in range(0, n_sims, CHUNK)
            ]
        )  # (n_sims, n_samples)
        truth = np.asarray(sims["b_group"])[:, 0]
        lower = {
            f"lower_cov_{1 - a:g}": float(
                (truth > np.quantile(draws, a, axis=1)).mean()
            )
            for a in ALPHAS
        } | {
            f"upper_cov_{1 - a:g}": float(
                (truth < np.quantile(draws, 1 - a, axis=1)).mean()
            )
            for a in ALPHAS
        }
        metric_rows.append(
            {
                **label,
                **compute_condition_metrics(draws, truth, cond_id, metric_fns),
                **lower,
            }
        )
        print(f"condition {cond_id + 1}/{len(val_data.simulations)}", flush=True)
        for c in THRESHOLDS:
            p_hat = (draws > c).mean(axis=1)
            for p, hit in zip(p_hat, truth > c, strict=True):
                rows.append({**label, "threshold": c, "p_hat": p, "hit": hit})
    df = pd.DataFrame(rows)
    df["bin"] = pd.cut(df["p_hat"], PROB_BINS, include_lowest=True)
    table = (
        df.groupby(["threshold", "bin"], observed=True)
        .agg(n=("hit", "size"), mean_p_hat=("p_hat", "mean"), observed=("hit", "mean"))
        .reset_index()
    )
    decisions = pd.DataFrame(
        [
            {
                "side": side,
                "threshold": c,
                "gamma": g,
                "n_decisions": int(hit.sum()),
                "mean_p_hat": float(sub.loc[hit, "p_hat"].mean()),
                "observed": float(sub.loc[hit, "hit"].mean()),
            }
            for c, sub in df.groupby("threshold")
            for g in GAMMAS
            for side, hit in [
                ("efficacy", sub["p_hat"] >= g),
                ("futility", sub["p_hat"] <= 1 - g),
            ]
        ]
    ).sort_values(["side", "threshold", "gamma"])
    return pd.DataFrame(metric_rows), table, decisions


def load_config(path: Path, **overrides) -> ANCOVAConfig:
    """Read a saved ``config.json`` (lists back to tuples), apply overrides."""
    saved = json.loads(path.read_text())
    saved = {k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()}
    return ANCOVAConfig(**(saved | overrides))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=ANCOVAConfig.epochs)
    parser.add_argument(
        "--batches-per-epoch", type=int, default=ANCOVAConfig.batches_per_epoch
    )
    # 250 for screening runs (about 35 min on the GTX 1660); 500 for final reports.
    parser.add_argument("--sims-per-condition", type=int, default=250)
    parser.add_argument("--posterior-samples", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--seed", type=int, default=2026)
    # Default 1/1 is pure log-uniform N; 0.7/0.7 is the v3 mixture.
    parser.add_argument("--n-log-share", type=float, default=ANCOVAConfig.n_log_share)
    parser.add_argument("--n-log-alpha", type=float, default=ANCOVAConfig.n_log_alpha)
    parser.add_argument("--integrate-method", default="tsit5")
    parser.add_argument("--integrate-steps", default="8")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="load <out>/approximator.keras (or .weights.h5); skip training",
    )
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    steps = args.integrate_steps
    solver = dict(
        inference_integrate_method=args.integrate_method,
        inference_integrate_steps=steps if steps == "adaptive" else int(steps),
    )
    if args.validate_only and (out / "config.json").exists():
        # Rebuild the trained architecture, not the current defaults.
        config = load_config(out / "config.json", **solver)
    else:
        config = ANCOVAConfig(
            epochs=args.epochs,
            batches_per_epoch=args.batches_per_epoch,
            n_log_share=args.n_log_share,
            n_log_alpha=args.n_log_alpha,
            **solver,
        )
    workflow = create_ancova_workflow(config, rng=np.random.default_rng(args.seed))
    tag = f"{args.integrate_method}_{steps}"

    if args.validate_only:
        adapted = workflow.adapter(workflow.simulator.sample(2))
        workflow.approximator.build_from_data(adapted)
        full = out / "approximator.keras"
        if full.exists():
            workflow.approximator.load_weights(full)
        else:
            workflow.approximator.load_weights(out / "approximator.weights.h5")
    else:
        train(workflow, config, out)

    val_data = build_validation_dataset(
        config, DEFAULT_CONDITION_GRID, sims_per_condition=args.sims_per_condition
    )
    t0 = time.perf_counter()
    per_cond, table, decisions = validate(
        workflow.approximator, val_data, args.posterior_samples
    )
    print(f"validation ({tag}): {time.perf_counter() - t0:.0f}s")
    per_cond.to_csv(out / f"sbc_per_condition_{tag}.csv", index=False)
    metrics = per_cond.drop(columns=list(DEFAULT_CONDITION_GRID))
    summary = aggregate_condition_rows(metrics.to_dict("records"))
    print("SBC summary:", summary)
    (out / f"sbc_summary_{tag}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    table.to_csv(out / f"tail_calibration_{tag}.csv", index=False)
    decisions.to_csv(out / f"decisions_{tag}.csv", index=False)
    side_cols = [c for c in per_cond if c.startswith(("lower_cov_", "upper_cov_"))]
    print(per_cond.groupby("N")[side_cols].mean().round(3).to_string())
    print(decisions.round(3).to_string(index=False))
    print(table.to_string(index=False))


def train(workflow, config: ANCOVAConfig, out: Path) -> None:
    """Train, then save the approximator (weights only if full save fails)."""
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2))
    t0 = time.perf_counter()
    history = train_ancova_workflow(workflow, config)
    train_s = time.perf_counter() - t0
    pd.DataFrame(history.history).to_csv(out / "history.csv", index=False)
    print(f"training: {train_s:.0f}s for {len(history.history['loss'])} epochs")

    try:
        workflow.approximator.save(out / "approximator.keras")
    except Exception as exc:  # adapter ufuncs may not serialize
        print(f"full save failed ({exc!r}); saving weights only")
        (out / "approximator.keras").unlink(missing_ok=True)
        workflow.approximator.save_weights(out / "approximator.weights.h5")


if __name__ == "__main__":
    main()
