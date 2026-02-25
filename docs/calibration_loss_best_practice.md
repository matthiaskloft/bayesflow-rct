# Calibration Loss Development Best Practice

This guide documents the recommended workflow for developing and stabilizing the new calibration loss in ANCOVA NPE training.

## Notebook Roles

- Development and debugging notebook: `examples/ancova_calibration_loss_development.ipynb`
- Comparison/reporting notebook: `examples/ancova_calibration_loss.ipynb`

Use the development notebook to test and tune settings. Keep the comparison notebook focused on clean baseline-vs-calibrated validation outputs.

## Recommended Development Workflow

1. **Environment and API sanity checks**
   - Verify `bayesflow_calibration_loss` package source and Lagrangian API kwargs on `CalibratedContinuousApproximator`.
   - Confirm required kwargs are present: `target_calibration_error`, `lr_lambda`, `lambda_max`, `normalization_burn_in`, `batch_size_calibration`.
   - Confirm Keras backend and package versions.

2. **Prior scale diagnostics**
   - Compare simulator prior draws, adapter-transformed inference variables, and `prior_fn` samples.
   - Ensure no severe scale mismatch before running sweeps.

3. **Coarse sweep (fast)**
   - Sweep `lambda_max` (or notebook compatibility alias `gamma_max`), `calibration_mode`, `n_samples`, and `batch_size_calibration`.
   - Tune `start_epoch` warmup and `normalization_burn_in` so dual updates are stable.
   - Include `target_calibration_error` and `lr_lambda` in sweep summaries for reproducibility.
   - Filter out unstable runs and rank by calibration quality with guardrails on NRMSE.

4. **Refinement sweep (robustness)**
   - Re-run top coarse candidates at larger training budgets.
   - Use multiple random seeds.
   - Select final settings by robust aggregate score (mean + variance penalties).

5. **Record output artifacts**
   - Export coarse and refinement results as CSV files.
   - Store a final recommended settings dictionary for downstream notebooks.

## Practical Defaults (Starting Point)

These are pragmatic starting values before project-specific tuning:

- `target_calibration_error`: `0.02`
- `lr_lambda`: `0.05`
- `lambda_max`: `0.5` to `2.0`
- `start_epoch`: `6` to `12`
- `normalization_burn_in`: approximately `0.5 × active_window_epochs × batches_per_epoch` (minimum `20`)
- `calibration_mode`: `0.0` (conservativeness mode)
- `n_samples`: `100` to `200`
- `batch_size_calibration`: `64` to `128` (or `None` for full batch, memory permitting)
- `normalize_loss`: `True`
- Use warmup (`start_epoch`) to avoid strong calibration pressure during random initialization.

## Acceptance Criteria for a Candidate Setting

A candidate is suitable if it meets all of the following:

- Improves or maintains calibration error relative to baseline.
- Does not cause unacceptable degradation in NRMSE.
- Trains stably across repeated seeds.
- Produces no systematic crashes or NaN losses during sweep/refinement.

## Maintenance Notes

- Re-run the full development notebook after major architecture changes or simulator changes.
- Keep this guide and the development notebook synchronized when the sweep logic or scoring changes.
