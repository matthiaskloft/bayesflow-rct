"""
ANCOVA Adapter Construction.

Builds the BayesFlow adapter for the ANCOVA 2-arms continuous outcome model.
"""

from __future__ import annotations

import keras
import numpy as np
from bayesflow import Adapter
from bayesflow.adapters.transforms import (
    ConvertDType,
    ElementwiseTransform,
    FilterTransform,
)

CONTEXT_KEYS = ["N", "p_alloc", "prior_df", "prior_scale"]
DATA_KEYS = ["outcome", "covariate", "group"]


@keras.saving.register_keras_serializable(package="bayesflow_rct")
class DatasetStandardize(ElementwiseTransform):
    """
    Standardize each dataset along its observation axis (mean 0, SD 1, ddof=1).

    Applied to raw ``(batch, N)`` arrays, identically for simulated and real
    data, so callers never preprocess. The treatment effect in residual-SD
    units (Cohen's d) is invariant to this affine transform of the outcome.
    ``inverse`` is the identity: data keys are never mapped back.
    """

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:  # noqa: D401
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, ddof=1, keepdims=True)
        if not np.all(std > 0):
            raise ValueError(
                "DatasetStandardize: a dataset has zero (or undefined) SD; "
                "outcome and covariate must vary and N must be >= 2."
            )
        return (data - mean) / std

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:  # noqa: D401
        return data

    def get_config(self) -> dict:
        return {}


def create_ancova_adapter() -> Adapter:
    """
    Create adapter for ANCOVA 2-arms model using the BayesFlow fluent API.

    The adapter pipeline:

    1. Standardize ``outcome`` and ``covariate`` per dataset
       (:class:`DatasetStandardize`).
    2. Broadcast scalar context (N, p_alloc, prior_df, prior_scale) to
       ``(batch, 1)`` using outcome's batch dim — before ``as_set`` makes
       outcome 3D, and targeting outcome so it also works at inference.
    3. Mark outcome/covariate/group as set-based data.
    4. Transform context: ``sqrt(N)``, ``log(prior_df)``, ``log(prior_scale)``.
    5. Map to canonical keys: ``b_group`` → ``inference_variables``, data →
       ``summary_variables``, context → ``inference_conditions``.
    6. Convert to float32.

    The approximator's own standardization handles inference variables and
    conditions.

    Returns
    -------
    Adapter configured for ANCOVA 2-arms model
    """
    adapter = Adapter()

    # Real data may carry integer arrays (e.g. group); cast the data arrays to
    # float64 so the final float64 -> float32 conversion covers them.
    adapter.add_transform(
        FilterTransform(
            include=DATA_KEYS,
            transform_constructor=ConvertDType,
            from_dtype="int64",
            to_dtype="float64",
        )
    )

    adapter.add_transform(
        FilterTransform(
            include=["outcome", "covariate"],
            transform_constructor=DatasetStandardize,
        )
    )

    for ctx_key in CONTEXT_KEYS:
        adapter.broadcast(ctx_key, to="outcome")

    adapter.as_set(DATA_KEYS)

    adapter.apply("N", forward=np.sqrt, inverse=np.square)
    adapter.apply(["prior_df", "prior_scale"], forward=np.log, inverse=np.exp)

    adapter.drop("b_covariate")

    adapter.rename("b_group", "inference_variables")
    adapter.concatenate(DATA_KEYS, into="summary_variables", axis=-1)
    adapter.concatenate(CONTEXT_KEYS, into="inference_conditions", axis=-1)

    adapter.convert_dtype("float64", "float32")

    return adapter
