"""
ANCOVA Model Configuration.

Flat dataclass holding all parameters for the ANCOVA 2-arms model.
HPO discovers the best architecture and saves results directly into
this config via ``ANCOVAConfig(**best_trial.params)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bayesflow_hpo as hpo
from bayesflow_hpo.builders import (
    build_inference_network,
    build_summary_network,
)


@dataclass
class ANCOVAConfig:
    """
    Complete ANCOVA model specification.

    All fields live flat on the dataclass, grouped by concern.
    After HPO, instantiate with ``ANCOVAConfig(**best_trial.params)``
    and serialize with ``dataclasses.asdict(config)``.

    Parameters
    ----------
    b_covariate_scale : float
        Scale for b_covariate Normal prior distribution.
    n_min, n_max : int
        Sample-size range for meta-parameter sampling.
    p_alloc_min, p_alloc_max : float
        Treatment allocation probability range.
    prior_df_min, prior_df_max : int
        Degrees-of-freedom range for the b_group prior.
    prior_df_alpha : float
        Alpha for log-uniform sampling of prior_df.
    prior_scale_gamma_shape, prior_scale_gamma_scale : float
        Gamma distribution parameters for prior_scale sampling.
    summary_dim : int
        Output dimensionality of the summary network.
    summary_depth : int
        Number of hidden layers in the summary network.
    summary_width : int
        Width of hidden layers in the summary network.
    summary_dropout : float
        Dropout rate for the summary network.
    inference_network_type : str
        ``"FlowMatching"`` or ``"CouplingFlow"``.
    inference_depth : int
        Number of flow layers / subnet depth.
    inference_widths : tuple of int
        Hidden-layer widths for FlowMatching subnets.
    inference_hidden_sizes : tuple of int
        Hidden-layer sizes for CouplingFlow subnets.
    inference_dropout : float
        Dropout rate for the inference network.
    inference_use_optimal_transport : bool
        Whether to use optimal transport in FlowMatching.
    initial_lr : float
        Initial learning rate.
    decay_rate : float
        Learning rate exponential decay factor.
    batch_size : int
        Training batch size.
    epochs : int
        Maximum training epochs.
    batches_per_epoch : int
        Batches per epoch for online training.
    validation_sims : int
        Number of simulations for online validation.
    early_stopping_patience : int
        Epochs without improvement before stopping.
    early_stopping_window : int
        Moving-average window for early stopping.
    """

    # --- Simulation / prior ---
    b_covariate_scale: float = 2.0
    n_min: int = 20
    n_max: int = 1000
    p_alloc_min: float = 0.5
    p_alloc_max: float = 0.9
    prior_df_min: int = 0
    prior_df_max: int = 30
    prior_df_alpha: float = 0.7
    prior_scale_gamma_shape: float = 2.0
    prior_scale_gamma_scale: float = 1.0

    # --- Summary network (discoverable by HPO) ---
    summary_dim: int = 10
    summary_depth: int = 3
    summary_width: int = 64
    summary_dropout: float = 0.05

    # --- Inference network (discoverable by HPO) ---
    inference_network_type: str = "FlowMatching"
    inference_depth: int = 7
    inference_widths: tuple[int, ...] = (125, 125, 125)
    inference_hidden_sizes: tuple[int, ...] = (128, 128)
    inference_dropout: float = 0.05
    inference_use_optimal_transport: bool = True

    # --- Training (discoverable by HPO) ---
    initial_lr: float = 7e-4
    decay_rate: float = 0.85
    batch_size: int = 320
    epochs: int = 200
    batches_per_epoch: int = 50
    validation_sims: int = 1000
    early_stopping_patience: int = 10
    early_stopping_window: int = 5


def hpo_params_to_config(params: dict[str, Any]) -> ANCOVAConfig:
    """
    Map HPO trial parameters to a flat :class:`ANCOVAConfig`.

    Optuna search spaces use prefixed keys (``ds_*``, ``cf_*``, ``fm_*``)
    that don't match ``ANCOVAConfig`` field names.  This helper translates
    them so that ``build_networks(hpo_params_to_config(trial.params))``
    produces the HPO-tuned architecture instead of falling back to defaults.

    Parameters
    ----------
    params : dict
        Flat dict from ``trial.params`` (Optuna) or any HPO output.
        Keys may be prefixed (``ds_summary_dim``, ``cf_depth``, …) or
        already match ``ANCOVAConfig`` fields (``initial_lr``, …).

    Returns
    -------
    ANCOVAConfig
        Config with HPO-tuned values; unrecognised keys are ignored.
    """
    defaults = ANCOVAConfig()
    cfg: dict[str, Any] = {}

    # --- Summary network (DeepSet space, ds_* prefix) ---
    if "ds_summary_dim" in params:
        cfg["summary_dim"] = int(params["ds_summary_dim"])
    if "ds_depth" in params:
        cfg["summary_depth"] = int(params["ds_depth"])
    if "ds_width" in params:
        cfg["summary_width"] = int(params["ds_width"])
    if "ds_dropout" in params:
        cfg["summary_dropout"] = float(params["ds_dropout"])

    # --- Inference network ---
    if "cf_depth" in params or "cf_subnet_width" in params:
        # CouplingFlow space (cf_* prefix)
        cfg["inference_network_type"] = "CouplingFlow"
        if "cf_depth" in params:
            cfg["inference_depth"] = int(params["cf_depth"])
        depth = int(
            params.get("cf_subnet_depth", len(defaults.inference_hidden_sizes))
        )
        width = int(
            params.get("cf_subnet_width", defaults.inference_hidden_sizes[0])
        )
        cfg["inference_hidden_sizes"] = tuple([width] * depth)
        if "cf_dropout" in params:
            cfg["inference_dropout"] = float(params["cf_dropout"])

    elif "fm_subnet_width" in params or "fm_subnet_depth" in params:
        # FlowMatching space (fm_* prefix)
        cfg["inference_network_type"] = "FlowMatching"
        depth = int(
            params.get("fm_subnet_depth", len(defaults.inference_widths))
        )
        width = int(
            params.get("fm_subnet_width", defaults.inference_widths[0])
        )
        cfg["inference_widths"] = tuple([width] * depth)
        cfg["inference_depth"] = depth
        if "fm_dropout" in params:
            cfg["inference_dropout"] = float(params["fm_dropout"])
        if "fm_use_ot" in params:
            cfg["inference_use_optimal_transport"] = bool(params["fm_use_ot"])

    # --- Training (no prefix, pass through directly) ---
    for key in ("initial_lr", "decay_rate", "batch_size", "epochs",
                "batches_per_epoch"):
        if key in params:
            cfg[key] = type(getattr(defaults, key))(params[key])

    return ANCOVAConfig(**cfg)


def build_networks(
    config: ANCOVAConfig,
) -> tuple:
    """
    Build summary and inference networks from a flat config.

    Uses ``bayesflow_hpo`` search-space objects to construct the networks
    from the config's architecture fields.

    Parameters
    ----------
    config : ANCOVAConfig
        Flat model specification.

    Returns
    -------
    tuple of (summary_net, inference_net)
    """
    summary_space = hpo.DeepSetSpace()
    summary_params: dict[str, Any] = {
        "ds_summary_dim": int(config.summary_dim),
        "ds_depth": int(config.summary_depth),
        "ds_width": int(config.summary_width),
        "ds_dropout": float(config.summary_dropout),
    }

    if config.inference_network_type == "FlowMatching":
        inference_space = hpo.FlowMatchingSpace()
        fm_width = int(config.inference_widths[0]) if config.inference_widths else 128
        inference_params: dict[str, Any] = {
            "fm_subnet_width": fm_width,
            "fm_subnet_depth": len(config.inference_widths) or 3,
            "fm_dropout": float(config.inference_dropout),
            "fm_activation": "mish",
            "fm_use_ot": bool(config.inference_use_optimal_transport),
        }
    elif config.inference_network_type == "CouplingFlow":
        inference_space = hpo.CouplingFlowSpace()
        cf_width = (
            int(config.inference_hidden_sizes[0])
            if config.inference_hidden_sizes
            else 128
        )
        inference_params = {
            "cf_depth": int(config.inference_depth),
            "cf_subnet_width": cf_width,
            "cf_subnet_depth": len(config.inference_hidden_sizes) or 2,
            "cf_dropout": float(config.inference_dropout),
            "cf_activation": "silu",
        }
    else:
        raise ValueError(
            f"Unknown inference network type: {config.inference_network_type}"
        )

    composite_space = hpo.CompositeSearchSpace(
        inference_space=inference_space,
        summary_space=summary_space,
    )
    params = {**summary_params, **inference_params}
    summary_net = build_summary_network(params=params, search_space=composite_space)
    inference_net = build_inference_network(params=params, search_space=composite_space)
    return summary_net, inference_net
