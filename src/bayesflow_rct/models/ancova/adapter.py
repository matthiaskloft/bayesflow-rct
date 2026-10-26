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
    Transform,
)

CONTEXT_KEYS = ["N", "p_alloc", "prior_df", "prior_scale"]
DATA_KEYS = ["outcome", "covariate", "group"]
STAT_KEYS = ["x_bar1", "y_bar1", "r_xy", "d_hat", "log_se_d_hat"]


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


@keras.saving.register_keras_serializable(package="bayesflow_rct")
class SufficientStatistics(Transform):
    """
    Add exact sufficient statistics of the normal ANCOVA as new keys.

    For ``y = a + d*g + b*x + e``, ``e ~ N(0, s^2)``, the cross-products of
    ``[1, g, x, y]`` are sufficient. After per-dataset standardization of
    ``y`` and ``x`` (``sum = 0``, ``sum of squares = N - 1``) and with
    ``N``/``n1`` known from the context, three numbers remain:

    - ``x_bar1``, ``y_bar1``: treatment-arm means of standardized x and y
    - ``r_xy``: correlation of x and y

    Two exact functions of these are added as easier-to-use encodings:

    - ``d_hat``: OLS group coefficient over residual SD (Cohen's d scale,
      invariant to the standardization)
    - ``log_se_d_hat``: log of its delta-method SE,
      ``sqrt(inv(X'X)_gg + d_hat^2 / (2 * (N - 3)))``

    Expects standardized ``outcome``/``covariate`` and 0/1 ``group`` of shape
    ``(batch, N)``; outputs ``(batch, 1)`` arrays. ``inverse`` drops nothing.
    Sources: see ``docs/ancova_prior_spec.md`` (sufficient statistics).
    """

    def forward(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        y, x, g = data["outcome"], data["covariate"], data["group"]
        n = y.shape[-1]
        n1 = g.sum(axis=-1, keepdims=True)
        design = np.stack([np.ones_like(g), g, x], axis=-1)  # (B, N, 3)
        xtx = np.einsum("bni,bnj->bij", design, design)
        xty = np.einsum("bni,bn->bi", design, y)
        beta = np.linalg.solve(xtx, xty[..., None])[..., 0]
        rss = ((y - np.einsum("bni,bi->bn", design, beta)) ** 2).sum(-1)
        dof = n - 3
        sigma = np.sqrt(rss / dof)
        d_hat = beta[:, 1] / sigma
        inv_gg = np.linalg.inv(xtx)[:, 1, 1]
        se = np.sqrt(inv_gg + d_hat**2 / (2 * dof))
        out = dict(data)
        out["x_bar1"] = (g * x).sum(-1, keepdims=True) / n1
        out["y_bar1"] = (g * y).sum(-1, keepdims=True) / n1
        out["r_xy"] = (x * y).sum(-1, keepdims=True) / (n - 1)
        out["d_hat"] = d_hat[:, None]
        out["log_se_d_hat"] = np.log(se)[:, None]
        return out

    def inverse(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        return data

    def get_config(self) -> dict:
        return {}


def create_ancova_adapter() -> Adapter:
    """
    Create adapter for ANCOVA 2-arms model using the BayesFlow fluent API.

    The adapter pipeline:

    1. Standardize ``outcome`` and ``covariate`` per dataset
       (:class:`DatasetStandardize`).
    2. Add exact sufficient statistics (:class:`SufficientStatistics`) as
       extra conditions, fused late (straight into the inference network).
    3. Broadcast scalar context (N, p_alloc, prior_df, prior_scale) to
       ``(batch, 1)`` using outcome's batch dim — before ``as_set`` makes
       outcome 3D, and targeting outcome so it also works at inference.
    4. Mark outcome/covariate/group as set-based data.
    5. Transform context: ``log(N)``, ``log(prior_df)``, ``log(prior_scale)``.
    6. Map to canonical keys: ``b_group`` → ``inference_variables``, data →
       ``summary_variables``, context → ``inference_conditions``.
    7. Convert to float32.

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

    adapter.add_transform(SufficientStatistics())

    for ctx_key in CONTEXT_KEYS:
        adapter.broadcast(ctx_key, to="outcome")

    adapter.as_set(DATA_KEYS)

    adapter.log("N")
    adapter.log(["prior_df", "prior_scale"])

    adapter.drop("b_covariate")

    adapter.rename("b_group", "inference_variables")
    adapter.concatenate(DATA_KEYS, into="summary_variables", axis=-1)
    adapter.concatenate(CONTEXT_KEYS + STAT_KEYS, into="inference_conditions", axis=-1)

    adapter.convert_dtype("float64", "float32")

    return adapter
