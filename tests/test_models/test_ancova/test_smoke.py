"""End-to-end smoke test: build -> train briefly -> validate."""

import numpy as np
import pytest

from bayesflow_rct.models.ancova.config import ANCOVAConfig
from bayesflow_rct.models.ancova.training import (
    create_ancova_workflow,
    train_ancova_workflow,
)
from bayesflow_rct.models.ancova.validation import (
    build_validation_dataset,
    run_validation,
)


@pytest.mark.slow
def test_train_and_validate_smoke():
    config = ANCOVAConfig(
        n_min=20,
        n_max=40,
        summary_dim=4,
        summary_width=16,
        summary_depth=1,
        inference_widths=(16, 16),
        epochs=2,
        batches_per_epoch=3,
        batch_size=16,
        lr_warmup_steps=1,
        validation_sims=16,
        early_stopping_window=1,
        early_stopping_patience=2,
    )
    workflow = create_ancova_workflow(config, rng=np.random.default_rng(1))
    history = train_ancova_workflow(workflow, config)
    assert "loss" in history.history

    val_data = build_validation_dataset(
        config,
        condition_grid={
            "N": [20, 40],
            "p_alloc": [0.5],
            "prior_df": [3.0],
            "prior_scale": [0.7],
        },
        sims_per_condition=8,
    )
    result = run_validation(workflow.approximator, val_data, n_posterior_samples=20)
    assert "calibration_error" in str(result.summary)
