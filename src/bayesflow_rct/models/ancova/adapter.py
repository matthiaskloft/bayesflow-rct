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
    3. Broadcast context scalars (N, p_alloc, prior_df, prior_scale)
       to match observation dimensions
    4. Apply transforms: sqrt(N), log1p(prior_df)
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

    # Broadcast scalar context variables to match observation dimensions
    for ctx_key in ["N", "p_alloc", "prior_df", "prior_scale"]:
        adapter.broadcast(ctx_key, to="outcome")

    # Apply transforms to context variables
    adapter.apply("N", forward=np.sqrt, inverse=np.square)
    adapter.apply("prior_df", forward=np.log1p, inverse=np.expm1)

    # Convert to float32
    adapter.convert_dtype("float64", "float32")

    return adapter
