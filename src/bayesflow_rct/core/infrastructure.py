"""Slim RCT infrastructure: ANCOVA simulator factory + PriorStandardize export."""

from collections.abc import Callable

import bayesflow as bf
from bayesflow_hpo.builders import PriorStandardize


def create_simulator(
    prior_fn: Callable,
    likelihood_fn: Callable,
    meta_fn: Callable | None = None,
) -> bf.simulators.Simulator:
    """Create a BayesFlow simulator from prior/likelihood/meta callables."""
    return bf.simulators.make_simulator([prior_fn, likelihood_fn], meta_fn=meta_fn)
