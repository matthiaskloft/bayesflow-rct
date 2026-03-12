# bayesflow-rct

Amortized Bayesian inference for Randomized Controlled Trials using [BayesFlow 2.x](https://github.com/bayesflow-org/bayesflow). Train neural posterior estimation (NPE) models once, then obtain instant posterior samples for any new RCT dataset — no MCMC required.

> **Note:** This package is experimental and under active development. APIs may change without notice.

## Features

- **ANCOVA model** for 2-arm continuous outcome trials with covariate adjustment
- **Hyperparameter optimization** via Optuna multi-objective search (calibration error vs. model size)
- **Simulation-based calibration (SBC)** validation across condition grids
- **Flexible architecture** — FlowMatching or CouplingFlow inference networks with DeepSet summary networks
- Built on [`bayesflow-hpo`](https://github.com/matthiaskloft/bayesflow-hpo) for generic HPO and validation infrastructure

## Installation

```bash
# Requires KERAS_BACKEND=torch
export KERAS_BACKEND=torch

# Basic install
pip install -e .

# With dev tools and notebooks
pip install -e ".[dev,notebooks]"

# With calibration loss support
pip install -e ".[calibration]"
```

Requires Python >= 3.11.

## Quick Start

```python
import bayesflow as bf
from bayesflow_rct import ANCOVAConfig, create_ancova_workflow_components
from bayesflow_rct.models.ancova.simulator import create_simulator

# Build components from config
config = ANCOVAConfig()
summary_net, inference_net, adapter = create_ancova_workflow_components(config)
simulator = create_simulator(config)

# Assemble and train
workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_net,
    summary_network=summary_net,
)

history = workflow.fit_online(epochs=config.epochs, batch_size=config.batch_size)
```

### Hyperparameter Optimization

```python
from bayesflow_rct import run_ancova_hpo

study = run_ancova_hpo(n_trials=50, study_name="ancova_hpo")
```

See `examples/` for full notebooks:
- `ancova_basic.ipynb` — Training and inference
- `ancova_optimization.ipynb` — Optuna HPO
- `ancova_calibration_loss.ipynb` — Training with calibration loss
## License

MIT
