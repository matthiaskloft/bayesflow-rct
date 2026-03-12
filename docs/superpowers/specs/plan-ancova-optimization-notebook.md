# Plan: ANCOVA Optimization Notebook Re-implementation

**Created**: 2026-03-11
**Author**: Claude
**Spec**: [2026-03-11-ancova-optimization-notebook-design.md](2026-03-11-ancova-optimization-notebook-design.md)

## Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Plan | DONE | 2026-03-11 | |
| Phase 1: Bug fixes | MERGED | 2026-03-12 | PR #9 |
| Phase 2: Notebook | MERGED | 2026-03-12 | PR #9 |
| Ship | MERGED | 2026-03-12 | PR #9 |

## Summary

**Motivation**: The current `ancova_optimization.ipynb` manually wires Optuna
studies, objectives, and train/validate loops — plumbing that `hpo.optimize()`
now handles. It also uses CouplingFlow (replaced by FlowMatching) and includes
threshold training that belongs in a deployment script.

**Outcome**: A streamlined ~12-cell notebook that finds the best FlowMatching
architecture via `hpo.optimize()`, analyzes the Pareto front, and exports the
best config as JSON for the deployment script. Three pre-existing issues are fixed
along the way.

## Assumptions

- `bayesflow-hpo` is installed in editable mode and its API is stable
- The notebook is for architecture search only; deep training lives elsewhere
- `hpo.optimize()` early stopping defaults (patience=5, window=7) are
  acceptable for HPO trials even though `ANCOVAConfig` uses different values
  (blocked by bayesflow-hpo Issue #6)

## Design Decisions

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Inference network | CouplingFlow vs FlowMatching | FlowMatching | User decision; better fits deployment pipeline |
| Optimal transport | Search over it vs fix True vs fix False | Fix True | Avoids doubling search space; OT is always-on in deployment |
| HPO entry point | Manual Optuna wiring vs `hpo.optimize()` | `hpo.optimize()` | Eliminates 4 cells of boilerplate |
| Validation data | Auto-generated vs pre-built | Pre-built `ValidationDataset` | ANCOVA condition grid is domain-specific; should be visible |
| Cost metric | `inference_time` (default) vs `param_count` | `param_count` | Matches old notebook; more interpretable for architecture search |
| Threshold training | Keep in notebook vs move to deployment | Move to deployment | Notebook's purpose is architecture search, not production training |
| Config export format | Raw trial params vs `ANCOVAConfig` JSON | `ANCOVAConfig` JSON with provenance | Deployment script can load directly; provenance enables traceability |

## Scope

### In Scope

- Fix `training.py:202` — stale dict access on `ValidationResult`
- Fix `config.py:hpo_params_to_config` — set `inference_depth` for FlowMatching
- Fix `model.py:__all__` — add missing `hpo_params_to_config` export
- Re-implement `ancova_optimization.ipynb` per the design spec
- Lint check (`ruff check src/`)

### Out of Scope

- Changes to `bayesflow-hpo` API (tracked as Issues #4, #6)
- Deployment script implementation
- `save_config()`/`load_config()` helpers in `config.py` (inline JSON in notebook is sufficient for now)
- Changes to other notebooks (`ancova_basic.ipynb`, `ancova_calibration_loss.ipynb`)

## Implementation Plan

### Phase 1: Bug fixes

**Files to create:**
- None

**Files to modify:**
- `src/bayesflow_rct/models/ancova/training.py` — fix line 202: `results["metrics"]` → `results.summary`
- `src/bayesflow_rct/models/ancova/config.py` — fix `hpo_params_to_config`: set `inference_depth` in the FlowMatching branch
- `src/bayesflow_rct/models/ancova/model.py` — add `hpo_params_to_config` to `__all__`

**Steps:**
1. In `training.py:202`, change `results["metrics"]` to `results.summary`
2. In `config.py`, FlowMatching branch of `hpo_params_to_config` (around line 169-182): add `cfg["inference_depth"] = depth` so the exported config reflects the actual subnet depth
3. In `model.py:__all__`, add `"hpo_params_to_config"` to the Config section (after `"build_networks"`)
4. Run `ruff check src/` to verify no lint errors introduced

**Depends on:** None

### Phase 2: Notebook

**Files to create:**
- None (overwriting existing notebook)

**Files to modify:**
- `examples/ancova_optimization.ipynb` — full re-implementation

**Steps:**
1. **Markdown cell**: Title + objectives (architecture search, Pareto optimization, config export)
2. **Code cell (Setup)**: Imports, KERAS_BACKEND, RNG, ANCOVAConfig, simulator, adapter. Comment explaining config's role.
3. **Markdown cell**: Validation dataset section header
4. **Code cell (Validation)**: `create_validation_grid(extended=True)`, load-or-create `ValidationDataset`, save to `data/ancova_hpo_validation/`
5. **Markdown cell**: Search space section header
6. **Code cell (Search space)**: `CompositeSearchSpace` with `FlowMatchingSpace` (OT fixed True), `DeepSetSpace`, `TrainingSpace`. Custom ranges per spec.
7. **Markdown cell**: Optimization section header + resume note + early stopping limitation
8. **Code cell (Optimize)**: `hpo.optimize()` call with all params from spec: `param_keys`, `data_keys`, `validation_data`, `search_space`, `inference_conditions=["N", "p_alloc", "prior_df", "prior_scale"]`, `epochs=config.epochs`, `batches_per_epoch=config.batches_per_epoch`, `cost_metric="param_count"`, `storage`, `study_name`, `resume=True`
9. **Code cell (Dashboard)**: Print Optuna Dashboard command
10. **Markdown cell**: Analysis section header
11. **Code cell (Analyze)**: `summarize_study`, `trials_to_dataframe`, `get_pareto_trials`, `plot_pareto_front`, `plot_param_importance`
12. **Markdown cell**: Export section header
13. **Code cell (Export)**: Extract best trial, `hpo_params_to_config`, save JSON with provenance metadata

**Depends on:** Phase 1 (the `hpo_params_to_config` fix affects the exported config)

## Verification & Validation

- **Automated**: `ruff check src/` passes; `mypy src/` passes (if configured)
- **Manual**:
  - Open notebook in Jupyter, verify all cells parse without syntax errors
  - Run Section 1 (Setup) + Section 2 (Validation) to confirm imports work and dataset generation succeeds
  - Verify the search space dimensions match the spec's deviation table
  - Confirm the `hpo.optimize()` call includes all required parameters

## Dependencies

- `bayesflow-hpo` (installed, editable mode)
- `bayesflow>=2.0`
- No new dependencies added

## Notes

_Living section — updated during implementation._

- bayesflow-hpo Issue #4: `param_keys`/`data_keys` redundancy when `validation_data` provided
- bayesflow-hpo Issue #6: `early_stopping_patience`/`early_stopping_window` not exposed in `optimize()`

## Review Feedback

Reviewed in 1 iteration. 8 findings (1 blocker, 4 warnings, 3 suggestions).

**Blocker (fixed):**
- `hpo_params_to_config` missing from `model.py.__all__` — added to Phase 1 scope

**Warnings (addressed):**
- `inference_depth` not set in FlowMatching branch — already in Phase 1
- `inference_conditions` param must be explicit in `hpo.optimize()` call — clarified in Phase 2 step 8
- Validation dataset total size (144 x 500 = 72k sims) undocumented — noted; will add comment in notebook

**Suggestions (noted):**
- Import path consistency (model.py vs config.py) — resolved by `__all__` fix
- Phase 1 scope was incomplete — now lists all 3 fixes
- Notebook cell count (~12-13) is approximate — acceptable for a plan
