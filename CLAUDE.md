# CLAUDE.md


## Purpose

Python package for training Neural Posterior Estimation (NPE) models using BayesFlow for Randomized Controlled Trial (RCT) Bayesian power analysis. Implements ANCOVA models for 2-arm continuous outcome trials and composes generic HPO functionality from `bayesflow-hpo`.

Key capabilities:
- Train neural networks to approximate posterior distributions for RCT parameters
- ANCOVA-specific HPO wrapper in `models/ancova/hpo.py` built on `bayesflow-hpo`
- Simulation-based calibration (SBC) validation via `bayesflow-hpo.validation`



## Workflow

Always work on a git worktree, not the main repository.

1. **Spec** - Understand the task and read relevant code first
2. **Plan** - Use plan mode to design the approach
3. **Draft** - Implement the changes
4. **Simplify** - Review and simplify the solution
5. **Update Tests** - Add or update tests for changed functionality
6. **Update Docs** - Update docstrings and "docs/" markdown files if applicable
7. **Test** - Run `pytest`
8. **Verify** - Run `ruff check src/`
9. **Quality Check** - Use a subagent to verify code quality and safety
10. **Learnings** - If problems occurred, note mistakes to avoid in the Learnings section below
11. **Commit & PR** - Commit changes and create a pull request
12. **Clean-up worktrees** - After merge, delete the worktree and branch


## Quick Commands

```bash
# Environment setup (requires KERAS_BACKEND=torch)
pip install -e ".[dev,notebooks]"            # Editable install with dev + notebook tools
pip install -e ".[calibration]"              # Optional: calibration loss addon

# Testing
pytest                                       # Run all tests
pytest tests/test_core                       # Run core tests only
pytest -v --cov=bayesflow_rct            # Verbose with coverage

# Code quality
ruff check src/                              # Linting
mypy src/                                    # Type checking

# Training (interactive)
jupyter notebook examples/                   # Start Jupyter for notebooks
```


## Project Structure

```
bayesflow-rct/
├── src/bayesflow_rct/              # Main package
│   ├── core/                           # RCT-specific infrastructure and utilities
│   │   ├── dashboard.py                # Training dashboard / monitoring
│   │   ├── threshold.py                # Threshold-based retraining loop
│   │   └── utils.py                    # ANCOVA utility helpers
│   ├── models/
│   │   └── ancova/                     # ANCOVA 2-arms continuous outcome model
│   │       ├── config.py               # Flat ANCOVAConfig dataclass + build_networks()
│   │       ├── simulator.py            # prior(), likelihood(), meta(), factory functions
│   │       ├── adapter.py              # BayesFlow adapter spec + fluent API builder
│   │       ├── validation.py           # Condition-grid SBC validation
│   │       ├── training.py             # Optuna objective, threshold training helpers
│   │       ├── metadata.py             # Model metadata utilities
│   │       ├── model.py                # Re-export facade for backward compatibility
│   │       └── hpo.py                  # ANCOVA-specific HPO wrapper
│   └── plotting/
│       └── diagnostics.py              # SBC diagnostic plots
│
├── tests/                              # Mirror structure of src/
│   ├── test_core/
│   ├── test_models/test_ancova/
│   └── test_plotting/
│
├── examples/                           # Jupyter notebooks
│   ├── ancova_basic.ipynb              # Basic ANCOVA training
│   ├── ancova_calibration_loss.ipynb   # Calibration loss training comparison
│   ├── ancova_calibration_loss_development.ipynb  # Calibration loss development/exploration
│   └── ancova_optimization.ipynb       # Optuna hyperparameter optimization
│
├── docs/                               # Design docs and guides
│
├── pyproject.toml                      # Package config and dependencies
└── requirements.txt                    # Core dependencies
```


## Reference

### Key dependencies
- `bayesflow>=2.0` - Neural posterior estimation framework
- `bayesflow-hpo>=0.1.0` - Generic BayesFlow HPO/search-space/validation package
- `keras>=3.9,<3.13` - Deep learning (backend: PyTorch)
- `optuna>=3.0` - Bayesian hyperparameter optimization
- `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`

### Architecture patterns
- **Thin application layer**: generic HPO/validation/builders live in `bayesflow-hpo`; this repo keeps ANCOVA-specific simulation, adapter, and threshold logic
- **Flat config per model**: each model has a single flat `@dataclass` (e.g. `ANCOVAConfig`) with all ~28 fields. HPO results map to config fields via `hpo_params_to_config(best_trial.params)`. Future models get their own flat config in `models/<name>/config.py`
- **Module-per-concern**: each model is split into focused modules (config, simulator, adapter, validation, training, metadata) with a `model.py` re-export facade for backward compat
- **Multi-objective optimization**: Optuna studies optimize (calibration_error, param_count) on a Pareto front
- **External calibration loss**: `bayesflow-calibration-loss` ([bayesflow-calibration-loss](https://github.com/matthiaskloft/bayesflow-calibration-loss)) is a separate repo, installed via `pip install -e ".[calibration]"`

### Conventions
- Type hints throughout (mypy configured but `ignore_errors = true` on most modules — gradual adoption)
- NumPy-style docstrings
- Ruff for linting (line length 88, Python 3.11+)
- Test structure mirrors `src/` layout


## Learnings / Things to avoid

- `bayesflow.Adapter.convert_dtype(from_dtype, to_dtype)` requires **both** positional args — not just one
- bayesflow-hpo `get_workflow_metadata()` stores validation under `"validation"` key, not `"validation_results"`
- bayesflow-hpo removed `AdapterSpec` class — use BayesFlow native `Adapter` fluent API instead
- bayesflow-hpo renamed `compute_metrics()` → `compute_condition_metrics()` and `summarize_metrics()` → `aggregate_condition_rows()`
- `run_condition_grid_validation` returns `{"condition_rows": [...], "summary": {...}}` — not the old `{"metrics": {...}}` format
- Plotting functions like `plot_sbc_diagnostics` expect per-simulation data from `run_validation_pipeline()`, not per-condition aggregates from `run_condition_grid_validation`
- HPO trial params use prefixed keys (`ds_*`, `cf_*`, `fm_*`) that don't match `ANCOVAConfig` field names — use `hpo_params_to_config()` to map them, never filter with `if k in __dataclass_fields__`
- Keras `ExponentialDecay.decay_steps` counts optimizer steps (batches), not samples — use `batches_per_epoch`, not `batch_size * batches_per_epoch`
- FlowMatching uses `inference_widths`, CouplingFlow uses `inference_depth` + `inference_hidden_sizes` — don't mix them up
- Keras is pinned to `<3.13` for BayesFlow 2.0.7 compatibility — don't bump without testing
