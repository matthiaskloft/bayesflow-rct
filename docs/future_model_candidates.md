# Future Model Candidates

Status: notes only, nothing implemented. Written 2026-10-17 after the ANCOVA
v2 model (Cohen's d, sufficient-statistic conditions).

## Selection criterion

Amortized NPE pays off when per-fit inference is slow or fragile and no small
set of exact sufficient statistics does the work. ANCOVA turned out to be easy
on the second point: three statistics plus N and n1 determine the posterior
(`docs/ancova_prior_spec.md`, section 5). The next model should be genuinely
harder.

## Candidates

| Model | Why estimation is hard | rctbayespower fit |
|---|---|---|
| **Weibull survival, 2-arm PH (recommended next)** | Unknown shape means no fixed-size sufficient statistics; information scales with events, not N; administrative censoring (accrual, interim timing) plus dropout; skewed, heavy-tailed posteriors at few events | Already exists in `models_survival.R` (brms Weibull), with `accrual.R`, `dropout.R` and interim functions; NPE would replace brms fits |
| Exponential survival | Easy: events and exposure per arm are sufficient | Same as above; useful as a sanity baseline only |
| Cluster-randomized binary (logistic GLMM) | Few clusters, ICC near 0 (funnel), non-conjugate; MCMC slow/unstable, Laplace/INLA inaccurate | High for CRT power; needs a nested cluster-then-patient summary network |
| Ordinal, proportional odds (e.g. mRS) | Many cutpoint nuisances, sparse categories | Medium |
| MMRM with dropout | Covariance nuisances, missing data, variable visits | High (primary analysis in many trials) |

## Carry-over from ANCOVA

- Estimand reparameterized (log hazard ratio, risk difference or similar),
  nuisance parameters marginalized out.
- Treatment-effect prior family as conditions (scaled t with df and scale).
- One-sided efficacy/futility validation (`scripts/train_baseline.py`), 500
  datasets per condition.
- Hybrid summary: DeepSet over raw observations plus late-fused statistics
  (for Weibull: events and exposure per arm, Cox log-HR estimate and SE; not
  sufficient, but a strong input).
- GPU training, fixed-step `tsit5`/8 sampling, no optimal transport.

All of this still needs source-backed priors and design decisions before
implementation.
