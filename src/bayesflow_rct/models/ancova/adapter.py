"""
ANCOVA Adapter Construction.

Builds the BayesFlow adapter for the ANCOVA 2-arms continuous outcome model.
"""

from __future__ import annotations

import numpy as np
from bayesflow import Adapter


def get_ancova_adapter_spec() -> dict:
    """
    Return the adapter specification for ANCOVA 2-arms model as a plain dict.

    This declaratively describes how ANCOVA data maps to the BayesFlow adapter:
    - Set-based data: outcome, covariate, group (per-observation)
    - Parameters to infer: b_group (treatment effect)
    - Context variables: N, p_alloc, prior_df, prior_scale
    - Standardization of outcome and covariate
    - Prior standardization of b_group by prior_scale
    - Broadcasting and transformations for context variables

    Returns
    -------
    dict
        Declarative specification for ANCOVA adapter.
    """
    return {
        "set_keys": ["outcome", "covariate", "group"],
        "param_keys": ["b_group"],
        "context_keys": ["N", "p_alloc", "prior_df", "prior_scale"],
        "standardize_keys": ["outcome", "covariate"],
        "prior_standardize": {"b_group": (None, "prior_scale")},
        "broadcast_specs": {
            "N": "outcome",
            "p_alloc": "outcome",
            "prior_df": "outcome",
            "prior_scale": "outcome",
        },
        "context_transforms": {
            "N": (np.sqrt, np.square),
            "prior_df": (np.log1p, np.expm1),
        },
        "output_dtype": "float32",
    }


def create_ancova_adapter() -> Adapter:
    """
    Create adapter for ANCOVA 2-arms model using the BayesFlow fluent API.

    The adapter pipeline:
    1. Broadcast scalar context (N, p_alloc, prior_df, prior_scale) to
       (batch, 1) using outcome's batch dim — must happen before as_set
       turns outcome 3D, and must target outcome (not b_group) so the
       key is available during both training and inference
    2. Mark outcome/covariate/group as set-based data (adds set dim)
    3. Standardize outcome and covariate (zero mean, unit variance)
    4. Apply transforms to context: sqrt(N), log1p(prior_df)
    5. Map to canonical BayesFlow keys:
       - b_group → inference_variables
       - outcome, covariate, group → summary_variables (3D set tensor)
       - N, p_alloc, prior_df, prior_scale → inference_conditions (2D)
    6. Convert all data to float32

    Returns
    -------
    Adapter configured for ANCOVA 2-arms model
    """
    adapter = Adapter()

    # Broadcast scalar context to (batch, 1) using outcome's batch dim.
    # Must happen BEFORE as_set (outcome is 2D here; after as_set it's 3D).
    # Uses outcome (not b_group) so the target key exists during inference.
    for ctx_key in ["N", "p_alloc", "prior_df", "prior_scale"]:
        adapter.broadcast(ctx_key, to="outcome")

    # Set-based data: each observation is one element of the set
    adapter.as_set(["outcome", "covariate", "group"])

    # Standardize observation-level data
    adapter.standardize(["outcome", "covariate"], mean=0.0, std=1.0)

    # Apply transforms to context variables
    adapter.apply("N", forward=np.sqrt, inverse=np.square)
    adapter.apply("prior_df", forward=np.log1p, inverse=np.expm1)

    # Drop nuisance parameter (not an inference target or condition)
    adapter.drop("b_covariate")

    # Map to canonical BayesFlow keys
    adapter.rename("b_group", "inference_variables")
    adapter.concatenate(
        ["outcome", "covariate", "group"], into="summary_variables", axis=-1
    )
    adapter.concatenate(
        ["N", "p_alloc", "prior_df", "prior_scale"],
        into="inference_conditions",
        axis=-1,
    )

    # Convert to float32
    adapter.convert_dtype("float64", "float32")

    return adapter
