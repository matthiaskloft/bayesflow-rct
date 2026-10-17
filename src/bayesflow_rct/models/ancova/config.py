"""
ANCOVA Model Configuration.

Flat dataclass holding all parameters for the ANCOVA 2-arms model.
HPO discovers the best architecture and saves results directly into
this config via ``ANCOVAConfig(**best_trial.params)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bayesflow as bf


@dataclass
class ANCOVAConfig:
    """
    Complete ANCOVA model specification.

    All fields live flat on the dataclass, grouped by concern.
    After HPO, instantiate with ``ANCOVAConfig(**best_trial.params)``
    and serialize with ``dataclasses.asdict(config)``.

    Parameters
    ----------
    n_min, n_max : int
        Log-uniform sample-size range.
    p_alloc_min, p_alloc_max : float
        Uniform range of the treatment allocation ratio.
    prior_df_median_shifted, prior_df_sdlog : float
        ``prior_df ~ 1 + LogNormal(log(median_shifted), sdlog)``.
    prior_scale_median, prior_scale_sdlog : float
        ``prior_scale ~ LogNormal(log(median), sdlog)``.
    d_bound : float
        Truncation bound ``|b_group| <= d_bound`` of the treatment-effect prior.
    r2_beta_a, r2_beta_b : float
        ``R2 ~ Beta(a, b)`` for the covariate's share of non-treatment variance.
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
        Peak learning rate (Adam), reached after ``lr_warmup_steps``.
    min_lr : float
        Final learning rate of the cosine decay.
    lr_warmup_steps : int
        Linear warmup in optimizer steps (batches).
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

    # --- Simulation / prior (docs/ancova_prior_spec.md) ---
    n_min: int = 20
    n_max: int = 1000
    p_alloc_min: float = 0.4
    p_alloc_max: float = 0.9
    prior_df_median_shifted: float = 5.0
    prior_df_sdlog: float = 1.0
    prior_scale_median: float = 0.7
    prior_scale_sdlog: float = 0.75
    d_bound: float = 5.0
    r2_beta_a: float = 1.2
    r2_beta_b: float = 2.0

    # --- Summary network (discoverable by HPO) ---
    summary_dim: int = 10
    summary_depth: int = 3
    summary_width: int = 64
    summary_dropout: float = 0.05

    # --- Inference network (discoverable by HPO) ---
    inference_network_type: str = "FlowMatching"
    inference_depth: int = 7
    inference_widths: tuple[int, ...] = (128, 128, 128)
    inference_hidden_sizes: tuple[int, ...] = (128, 128)
    inference_dropout: float = 0.05
    inference_use_optimal_transport: bool = True

    # --- Training (discoverable by HPO) ---
    initial_lr: float = 1e-3
    min_lr: float = 1e-4
    lr_warmup_steps: int = 25
    batch_size: int = 256
    epochs: int = 50
    batches_per_epoch: int = 100
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
    for key in ("initial_lr", "min_lr", "lr_warmup_steps", "batch_size",
                "epochs", "batches_per_epoch"):
        if key in params:
            cfg[key] = type(getattr(defaults, key))(params[key])

    return ANCOVAConfig(**cfg)


def build_networks(
    config: ANCOVAConfig,
) -> tuple:
    """
    Build summary and inference networks from a flat config.

    Parameters
    ----------
    config : ANCOVAConfig
        Flat model specification.

    Returns
    -------
    tuple of (summary_net, inference_net)
    """
    width = int(config.summary_width)
    mlp = (width,) * int(config.summary_depth)
    summary_net = bf.networks.DeepSet(
        summary_dim=int(config.summary_dim),
        mlp_widths_equivariant=mlp,
        mlp_widths_invariant_inner=mlp,
        mlp_widths_invariant_outer=mlp,
        mlp_widths_invariant_last=mlp,
        dropout=float(config.summary_dropout),
    )

    if config.inference_network_type == "FlowMatching":
        inference_net = bf.networks.FlowMatching(
            subnet_kwargs={
                "widths": tuple(int(w) for w in config.inference_widths),
                "dropout": float(config.inference_dropout),
            },
            use_optimal_transport=bool(config.inference_use_optimal_transport),
        )
    elif config.inference_network_type == "CouplingFlow":
        inference_net = bf.networks.CouplingFlow(
            depth=int(config.inference_depth),
            subnet_kwargs={
                "widths": tuple(int(w) for w in config.inference_hidden_sizes),
                "dropout": float(config.inference_dropout),
            },
        )
    else:
        raise ValueError(
            f"Unknown inference network type: {config.inference_network_type}"
        )
    return summary_net, inference_net
