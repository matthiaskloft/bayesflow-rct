# Design: `ancova_optimization.ipynb` Re-implementation

**Date:** 2026-03-11
**Status:** Approved
**Goal:** Re-implement the ANCOVA HPO notebook using the current `bayesflow-hpo` API (`hpo.optimize()`). The notebook finds the best FlowMatching architecture via Pareto-optimal multi-objective search. Deep training happens in a separate deployment script.

## Context

The current notebook manually wires Optuna studies, objectives, and the train/validate loop — plumbing that `hpo.optimize()` now handles. It also includes threshold-based retraining (cells 14–18) that belongs in the deployment script, not the search notebook.

### Key API: `bayesflow-hpo`

- `hpo.optimize()` — single-call HPO (study creation, objective, optimization loop)
- `hpo.FlowMatchingSpace` — search space for `bf.networks.FlowMatching`
- `hpo.DeepSetSpace` — search space for `bf.networks.DeepSet`
- `hpo.CompositeSearchSpace` — combines inference + summary + training spaces
- `hpo.ValidationDataset` — immutable, saveable/loadable validation data
- `hpo.get_pareto_trials()`, `hpo.trials_to_dataframe()`, `hpo.summarize_study()` — analysis
- `hpo.plot_pareto_front()` — visualization
- `hpo.save_validation_dataset()` / `hpo.load_validation_dataset()` — persistence

### Key API: `bayesflow-rct`

- `ANCOVAConfig` — flat dataclass with all ~28 hyperparameter fields
- `hpo_params_to_config(trial.params)` — maps prefixed Optuna keys to config fields
- `create_simulator(config, rng)` — ANCOVA simulator factory
- `create_ancova_adapter()` — BayesFlow adapter for ANCOVA data
- `create_validation_grid(extended=True)` — ANCOVA-specific condition grid
- `build_validation_dataset(conditions, n_sims, rng)` — builds `hpo.ValidationDataset` from ANCOVA conditions

## Design Decisions

1. **FlowMatching only** — no CouplingFlow. The search space uses `FlowMatchingSpace`.
2. **Optimal transport always on** — `fm_use_ot` fixed to `True` via single-value `CategoricalDimension`. This ensures architecture search happens under the same training dynamics as deployment.
3. **No threshold training** — removed from notebook. That's the deployment script's job.
4. **No model saving** — no trained model to save. The notebook exports the best *config*, not weights.
5. **Pre-built ValidationDataset** — ANCOVA condition grid is domain-specific and should be visible in the notebook, not buried inside `hpo.optimize()`.
6. **Dual output** — study DB (SQLite) for deep analysis + JSON config file for the deployment script.
7. **Cost metric: `param_count`** — the second Pareto objective is normalized parameter count (matching the old notebook), not the `hpo.optimize()` default `"inference_time"`.

## Notebook Structure

### Section 1: Setup (~2 cells)

```python
import os
if not os.environ.get("KERAS_BACKEND"):
    os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
import numpy as np
import bayesflow_hpo as hpo
from bayesflow_rct.models.ancova.model import (
    ANCOVAConfig,
    create_simulator,
    create_ancova_adapter,
    create_validation_grid,
    hpo_params_to_config,
)
from bayesflow_rct.models.ancova.validation import build_validation_dataset

RNG = np.random.default_rng(2025)

# Simulation defaults (prior ranges, N range, etc.) and training hyperparameters
# (epochs, batch_size, early stopping). The *architecture* fields will be overridden
# by HPO — the config exported in Section 6 is the final optimized version.
config = ANCOVAConfig()
simulator = create_simulator(config, RNG)
adapter = create_ancova_adapter()
```

### Section 2: Validation Dataset (~2 cells)

Build ANCOVA-specific `ValidationDataset` from the condition grid and save for reproducibility.

```python
conditions = create_validation_grid(extended=True)

val_data_path = Path("data/ancova_hpo_validation")
if val_data_path.exists():
    val_data = hpo.load_validation_dataset(val_data_path)
    print(f"Loaded cached validation dataset ({len(val_data.simulations)} conditions)")
else:
    val_data = build_validation_dataset(conditions, n_sims=500, rng=RNG)
    hpo.save_validation_dataset(val_data, val_data_path)
    print(f"Generated and saved validation dataset ({len(conditions)} conditions)")
```

The extended grid crosses: n_total (20, 200, 1000) x prior_df (0, 2) x prior_scale (0.1, 5.0) x b_covariate (-1.0, 1.0) x b_arm_treat (0.0, 0.3, 1.0) x p_alloc (0.5, 0.9) = 144 conditions.

### Section 3: Search Space (~1 cell)

```python
search_space = hpo.CompositeSearchSpace(
    inference_space=hpo.FlowMatchingSpace(
        subnet_width=hpo.IntDimension("fm_subnet_width", 32, 256, step=32),
        subnet_depth=hpo.IntDimension("fm_subnet_depth", 1, 4),
        dropout=hpo.FloatDimension("fm_dropout", 0.0, 0.2),
        use_optimal_transport=hpo.CategoricalDimension("fm_use_ot", choices=[True]),
    ),
    summary_space=hpo.DeepSetSpace(
        summary_dim=hpo.IntDimension("ds_summary_dim", 4, 16),
        width=hpo.IntDimension("ds_width", 32, 128, step=16),
        depth=hpo.IntDimension("ds_depth", 1, 4),
        dropout=hpo.FloatDimension("ds_dropout", 0.05, 0.5),
    ),
    training_space=hpo.TrainingSpace(
        initial_lr=hpo.FloatDimension("initial_lr", 1e-5, 5e-3, log=True),
        batch_size=hpo.IntDimension("batch_size", 128, 832, step=32),
    ),
)
```

Dimensions searched: `fm_subnet_width`, `fm_subnet_depth`, `fm_dropout`, `ds_summary_dim`, `ds_width`, `ds_depth`, `ds_dropout`, `initial_lr`, `batch_size` (9 tunable + 1 fixed).

**Deliberate deviations from `bayesflow-hpo` defaults:**

| Dimension | Default range | Spec range | Rationale |
|---|---|---|---|
| `ds_width` | 32–256, step 32 | 32–128, step 16 | Narrower search; ANCOVA has only 2 params, large summary nets are wasteful. Finer step for better resolution. |
| `ds_dropout` | 0.0–0.3 | 0.05–0.5 | Floor avoids no-dropout configs; higher ceiling allows stronger regularization for small-N conditions. |
| `initial_lr` | 1e-4–5e-3 | 1e-5–5e-3 | Extended lower bound to explore slower learning rates with OT-smoothed gradients. |
| `batch_size` | 32–1024, step 32 (off by default) | 128–832, step 32 (on) | Explicitly enabled; narrowed to avoid extreme sizes. |

### Section 4: Run Optimization (~2 cells)

```python
study = hpo.optimize(
    simulator=simulator,
    adapter=adapter,
    param_keys=["b_group"],
    data_keys=["outcome", "covariate", "group"],
    validation_data=val_data,
    search_space=search_space,
    inference_conditions=["N", "p_alloc", "prior_df", "prior_scale"],
    n_trials=100,
    epochs=config.epochs,
    batches_per_epoch=config.batches_per_epoch,
    cost_metric="param_count",
    storage="sqlite:///ancova_hpo.db",
    study_name="ancova_flowmatching",
    resume=True,
)
```

Replaces 4 cells in the current notebook (manual study creation, objective wiring, dashboard hint, study.optimize). A second cell prints the Optuna Dashboard command.

> **Note:** `resume=True` means re-running this cell continues from where it left off rather than starting a new study. Delete the SQLite DB to start fresh.

> **Known limitation:** `hpo.optimize()` does not expose `early_stopping_patience`/`early_stopping_window`, so trials use `ObjectiveConfig` defaults (patience=5, window=7) instead of `ANCOVAConfig` values (patience=10, window=5). See Issue #5.

### Section 5: Analyze Pareto Front (~3 cells)

```python
# Text summary
print(hpo.summarize_study(study))

# DataFrame
df = hpo.trials_to_dataframe(study)
pareto = hpo.get_pareto_trials(study)
pareto_df = df[df["trial_number"].isin({t.number for t in pareto})]
display(pareto_df.sort_values("objective_0"))

# Plots
hpo.plot_pareto_front(study)
hpo.plot_param_importance(study)
```

### Section 6: Export Best Config (~2 cells)

```python
from dataclasses import asdict
import json

best_trial = sorted(pareto, key=lambda t: t.values[0])[0]
config_best = hpo_params_to_config(best_trial.params)

export = {
    "config": asdict(config_best),
    "provenance": {
        "trial_number": best_trial.number,
        "objectives": list(best_trial.values),
        "study_name": "ancova_flowmatching",
        "study_db": "ancova_hpo.db",
    },
}

config_path = Path("configs/ancova_best.json")
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(json.dumps(export, indent=2, default=str))
```

## What's Removed vs. Current Notebook

| Current notebook | Re-implementation |
|---|---|
| Manual `optuna.create_study()` | Handled by `hpo.optimize()` |
| Manual `create_ancova_objective()` | Handled by `hpo.optimize()` |
| `study.optimize(objective, ...)` | Handled by `hpo.optimize()` |
| `CouplingFlowSpace` | Replaced with `FlowMatchingSpace` |
| Threshold training (cells 14–18) | Removed — deployment script |
| `save_model_with_metadata()` | Removed — no trained model |

## What's Added

| New | Why |
|---|---|
| `hpo.save_validation_dataset()` | Reproducibility |
| `hpo.summarize_study()` | Cleaner text summary |
| JSON config export | Hand-off to deployment script |

## New Code Needed

### `config.py`: `save_config()` / `load_config()` helpers (optional)

Simple JSON serialization of `ANCOVAConfig` via `dataclasses.asdict()`. Could be done inline in the notebook (as shown above) or extracted to a helper if the deployment script also needs it.

## Known Issues

### In `bayesflow-rct`

- **`training.py:202`** — `results["metrics"]` should be `results.summary`. `run_validation_pipeline()` returns a `ValidationResult` dataclass, not a dict. This is in the threshold training code (not used by this notebook) but is broken.
- **`config.py:hpo_params_to_config`** — FlowMatching branch sets `inference_widths` but not `inference_depth`, so the exported config retains the default `inference_depth=7` regardless of the HPO-discovered `fm_subnet_depth`. Not a runtime issue (since `build_networks()` reads `inference_widths`, not `inference_depth`, for FlowMatching) but the exported JSON is misleading.

### In `bayesflow-hpo`

- **Issue #4** — `optimize()` requires `param_keys`/`data_keys` even when `validation_data` (which embeds them) is provided. Filed: https://github.com/matthiaskloft/bayesflow-hpo/issues/4
- **Issue #5** — `optimize()` does not expose `early_stopping_patience` or `early_stopping_window`. These default to `patience=5, window=7` on `ObjectiveConfig`, which differs from `ANCOVAConfig` defaults (`patience=10, window=5`). Until resolved, HPO trials use the `ObjectiveConfig` defaults. Filed: https://github.com/matthiaskloft/bayesflow-hpo/issues/5

## Cell Count

Current notebook: 19 cells (9 code, 5 markdown, 5 output-heavy)
Re-implementation: ~12 cells (6 code, 4 markdown, 2 output)
