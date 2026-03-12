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
    1. Mark outcome/covariate/group as set-based data
    2. Standardize outcome and covariate (zero mean, unit variance)
    3. Apply transforms to context: sqrt(N), log1p(prior_df)
    4. Map to canonical BayesFlow keys:
       - b_group → inference_variables
       - outcome, covariate, group → summary_variables (3D set tensor)
       - N, p_alloc, prior_df, prior_scale → inference_conditions (2D)
    5. Convert all data to float32

    Returns
    -------
    Adapter configured for ANCOVA 2-arms model
    """
    adapter = Adapter()

    # Set-based data: each observation is one element of the set
    adapter.as_set(["outcome", "covariate", "group"])

    # Standardize observation-level data
    adapter.standardize(["outcome", "covariate"], mean=0.0, std=1.0)

    # Broadcast scalar context to (batch, 1) to match b_group shape,
    # then apply transforms (must broadcast before concat)
    for ctx_key in ["N", "p_alloc", "prior_df", "prior_scale"]:
        adapter.broadcast(ctx_key, to="b_group")
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
