"""
ANCOVA Metadata Utilities.

Helpers for collecting and persisting ANCOVA model metadata.
"""

from __future__ import annotations

from dataclasses import asdict

from bayesflow_hpo.results import (
    get_workflow_metadata,
    save_workflow_with_metadata,
)

from bayesflow_rct.models.ancova.config import ANCOVAConfig


def get_model_metadata(
    config: ANCOVAConfig,
    validation_results: dict = None,
    extra: dict = None,
) -> dict:
    """
    Collect all ANCOVA-specific reproducibility metadata.

    Parameters
    ----------
    config : ANCOVAConfig
        Configuration used for training.
    validation_results : dict, optional
        Validation metrics to include.
    extra : dict, optional
        Additional metadata to include.

    Returns
    -------
    dict with complete metadata
    """
    return get_workflow_metadata(
        config=asdict(config),
        model_type="ancova_cont_2arms",
        validation_results=validation_results,
        extra=extra,
    )


# Convenience alias used in notebooks
save_model_with_metadata = save_workflow_with_metadata
