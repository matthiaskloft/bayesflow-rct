"""
Threshold-Based Training Loop.

Retry-loop training that builds, trains, and validates models
until quality thresholds are met or max iterations are exhausted.

This module is independent of Optuna and can be used standalone.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================


@dataclass
class QualityThresholds:
    """
    Quality thresholds for convergence checking.

    Training continues until all thresholds are met or max_iterations reached.

    Attributes
    ----------
    max_cal_error : float
        Maximum mean calibration error (default: 0.02).
    max_c2st_deviation : float
        Maximum |C2ST - 0.5| (default: 0.05).
    max_coverage_error : float
        Maximum coverage deviation from nominal (default: 0.03).
    max_iterations : int
        Maximum training attempts (default: 10).
    min_improvement : float
        Minimum improvement to continue (default: 0.001).
    """

    max_cal_error: float = 0.02
    max_c2st_deviation: float = 0.05
    max_coverage_error: float = 0.03
    max_iterations: int = 10
    min_improvement: float = 0.001


# =============================================================================
# THRESHOLD CHECKING
# =============================================================================


def check_thresholds(
    metrics: dict,
    thresholds: QualityThresholds,
) -> tuple[bool, dict[str, float]]:
    """
    Check if metrics meet quality thresholds.

    Parameters
    ----------
    metrics : dict
        Metrics from validation pipeline with 'summary' key.
    thresholds : QualityThresholds
        Threshold values to check against.

    Returns
    -------
    tuple of (bool, dict)
        (passed, scores) where scores has individual metric values.
    """
    summary = metrics.get("summary", metrics)

    cal_error = summary.get("mean_cal_error", 1.0)
    c2st = summary.get("sbc_c2st_accuracy", 0.5)
    c2st_deviation = abs(c2st - 0.5)

    # Coverage at 95% level
    cov_95 = summary.get("coverage_95", 0.95)
    coverage_error = abs(cov_95 - 0.95)

    scores = {
        "cal_error": cal_error,
        "c2st_deviation": c2st_deviation,
        "coverage_error": coverage_error,
    }

    passed = (
        cal_error <= thresholds.max_cal_error
        and c2st_deviation <= thresholds.max_c2st_deviation
        and coverage_error <= thresholds.max_coverage_error
    )

    return passed, scores


# =============================================================================
# THRESHOLD-BASED TRAINING LOOP
# =============================================================================


def _cleanup() -> None:
    """Clean up GPU memory after a trial completes."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except (ImportError, AttributeError):
        pass


def train_until_threshold(
    build_workflow_fn: Callable[[dict], Any],
    train_fn: Callable[[Any], Any],
    validate_fn: Callable[[Any], dict],
    hyperparams: dict,
    thresholds: QualityThresholds = None,
    checkpoint_path: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Train a model repeatedly until quality thresholds are met.

    This function implements a retry loop that:
    1. Builds a fresh model with the given hyperparameters
    2. Trains the model
    3. Validates on a strict evaluation grid
    4. Checks if thresholds are met
    5. If not met and improving, retrain from scratch

    Parameters
    ----------
    build_workflow_fn : callable
        Function(hyperparams) -> workflow/model.
    train_fn : callable
        Function(workflow) -> history.
    validate_fn : callable
        Function(workflow) -> metrics_dict.
    hyperparams : dict
        Hyperparameters from best Optuna trial.
    thresholds : QualityThresholds, optional
        Quality thresholds. Default: QualityThresholds().
    checkpoint_path : str, optional
        Path to save best model checkpoint.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict
        Keys: 'workflow', 'metrics', 'history', 'iterations',
        'converged', 'best_scores'.
    """
    if thresholds is None:
        thresholds = QualityThresholds()

    best_workflow = None
    best_metrics = None
    best_composite = float("inf")
    best_scores = None
    all_histories: list[Any] = []
    converged = False

    for iteration in range(1, thresholds.max_iterations + 1):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Training Iteration {iteration}/{thresholds.max_iterations}")
            print(f"{'=' * 60}")

        # Build fresh model
        workflow = build_workflow_fn(hyperparams)

        # Train
        try:
            history = train_fn(workflow)
            all_histories.append(history)
        except Exception as e:
            if verbose:
                print(f"Training failed: {e}")
            _cleanup()
            continue

        # Validate on strict grid
        try:
            metrics = validate_fn(workflow)
        except Exception as e:
            if verbose:
                print(f"Validation failed: {e}")
            _cleanup()
            continue

        # Check thresholds
        passed, scores = check_thresholds(metrics, thresholds)

        # Compute composite score for comparison
        composite = (
            scores["cal_error"] + scores["c2st_deviation"] + scores["coverage_error"]
        )

        if verbose:
            print(f"\nIteration {iteration} Results:")
            print(
                f"  Calibration Error: {scores['cal_error']:.4f}"
                f" (threshold: {thresholds.max_cal_error})"
            )
            print(
                f"  C2ST Deviation:    {scores['c2st_deviation']:.4f}"
                f" (threshold: {thresholds.max_c2st_deviation})"
            )
            print(
                f"  Coverage Error:    {scores['coverage_error']:.4f}"
                f" (threshold: {thresholds.max_coverage_error})"
            )
            print(f"  Composite Score:   {composite:.4f}")
            status = "\u2713 YES" if passed else "\u2717 NO"
            print(f"  Passed: {status}")

        # Track best
        if composite < best_composite:
            improvement = best_composite - composite
            best_composite = composite
            best_workflow = workflow
            best_metrics = metrics
            best_scores = scores

            if verbose and iteration > 1:
                print(f"  \u2192 New best! Improvement: {improvement:.4f}")

            # Save checkpoint
            if checkpoint_path is not None:
                try:
                    if hasattr(workflow, "approximator"):
                        workflow.approximator.save(checkpoint_path)
                    else:
                        workflow.save(checkpoint_path)
                    if verbose:
                        print(f"  \u2192 Checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    if verbose:
                        print(f"  \u2192 Checkpoint save failed: {e}")
        else:
            improvement = best_composite - composite

        # Check convergence
        if passed:
            converged = True
            if verbose:
                print(f"\n\u2713 Thresholds met at iteration {iteration}!")
            break

        # Check if still improving enough to continue
        if iteration > 1 and improvement < thresholds.min_improvement:
            if verbose:
                print(
                    f"\n\u26a0 Insufficient improvement"
                    f" ({improvement:.4f}"
                    f" < {thresholds.min_improvement})"
                )
                print("  Consider adjusting architecture or training settings.")

        # Cleanup for next iteration
        if workflow is not best_workflow:
            del workflow
        _cleanup()

    if not converged and verbose:
        print("\n\u26a0 Max iterations reached without meeting thresholds.")
        print(f"  Best composite score: {best_composite:.4f}")
        print(f"  Best individual scores: {best_scores}")

    return {
        "workflow": best_workflow,
        "metrics": best_metrics,
        "history": all_histories,
        "iterations": len(all_histories),
        "converged": converged,
        "best_scores": best_scores,
    }
