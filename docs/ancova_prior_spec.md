# ANCOVA Prior Specification (draft)

Status: **implemented in Phase 0** (`models/ancova/simulator.py`, `adapter.py`). Values marked *(TBD)* still need
a source or a decision. Per project rules, every prior must be backed by a reference
(collected in `references.md`) before implementation.

## 1. Model

Simulate on the natural scale (residual SD = 1), then standardize the simulated data
exactly as real data are preprocessed. The network sees only standardized data and
estimates a single target, Cohen's `d`.

```r
# --- design (conditions, observed) ---
N        ~ log-uniform-int(20, 1000)
p_alloc  ~ Uniform(0.4, 0.9)
group    ~ exactly round(N * p_alloc) treated (fixed allocation)

# --- meta-parameters of the target prior (conditions) ---
prior_df    ~ 1 + LogNormal(log(5), 1.0)      # Student-t df >= 1; ~18% below 3, ~4% above 30
prior_scale ~ LogNormal(log(0.7), 0.75)         # median = JZS default 0.707

# --- target ---
d        ~ prior_scale * t(prior_df), truncated to |d| <= 5   # Cohen's d (residual-SD units)
# sampled by inverse CDF in float64:
#   u ~ U(F(-5/s), F(5/s)),  d = s * F^-1(u),  F = t CDF with prior_df, s = prior_scale

# --- nuisance (simulated, never estimated -> marginalized) ---
R2       ~ Beta(1.2, 2.0)                          # covariate share of non-treatment variance
b_cov    = sign * sqrt(R2 / (1 - R2)),  sign ~ ±1 with prob 0.5
x        ~ Normal(0, 1)

# --- likelihood ---
y_raw    = d * group + b_cov * x + Normal(0, 1)

# --- preprocessing (in the adapter, identical for training and inference) ---
y = (y_raw - mean(y_raw)) / sd(y_raw)
x = (x - mean(x)) / sd(x)
```

Network inputs: `outcome = y`, `covariate = x`, `group`, `N`, `p_alloc`,
`prior_df`, `prior_scale`. Output: posterior of `d` (key `b_group` / `b_arm_treat`).

## 2. Prior table

| Quantity | Role | Prior | Rationale | Status |
|---|---|---|---|---|
| `d` | target | `prior_scale · t(prior_df)`, truncated to \|d\| ≤ 5 | meta-amortized over the analysis priors users set in rctbayespower (default `student_t(3, 0, 1)`). Cap removes irrelevant extreme draws from heavy-tailed priors (Cauchy, scale 0.7: ~4% have \|d\| > 10); the network therefore implements the user prior truncated at ±5 — document this for rctbayespower. Inverse-CDF sampling: exact, no rejection, ~3.5 µs/draw | keep |
| `prior_df` | condition | 1 + LogNormal(log 5, 1.0) | log-scale, shifted so df ≥ 1 (no undefined-mean priors; Cauchy df = 1 is the boundary): 95% 1.7–36.5, P(df < 3) ≈ 18% (covers rctbayespower default df = 3), P(df < 2, infinite variance) ≈ 5%, P(df > 30) ≈ 4% (≈ Normal, no separate Normal switch). Chosen over Gamma(2, 0.1) (Juárez & Steel, 2010), which has only ~4% below df = 3. (TBD) verify source | review |
| `prior_scale` | condition | LogNormal(log 0.7, 0.75) | median at JZS default √2/2 (Rouder et al., 2009); 95%: 0.16–3.0 covers sceptical to ultrawide (√2); P(>4) ≈ 1%. (TBD) verify sources | review |
| `R2` (covariate) | nuisance | Beta(1.2, 2) | right-skewed (mode 0.17, mean 0.37), broad: 5–95% 0.04–0.80, P(R² > 0.9) ≈ 1.3%. Chosen for now; revisit after first training run. With N ≥ 20 the data identify R², so prior influence on the marginal posterior of d is small. (TBD) source for typical baseline–outcome R² in RCTs | review |
| intercept, σ | — | removed by standardization | d is invariant to affine transforms of y | — |
| `N` | condition | log-uniform 20–1000 | current config | keep |
| `p_alloc` | condition | Uniform(0.4, 0.9) | widened below 0.5 so imbalanced arms (more control than treatment, e.g. from Bernoulli assignment in a power scenario) are in the training range | decided |

## 3. Open questions

1. **Preprocessing contract.** Decided: standardization happens inside the
   adapter (same formula, `sd` with n−1, in training and inference). Callers such as
   rctbayespower pass raw data.
2. **MCID handling.** Decided: no shift of the data. The prior on d stays centered
   at 0; the MCID (rctbayespower `thr_fx_eff`) enters only when computing
   P(d > MCID) from the posterior draws. (Shifting the data would silently center
   each user's prior on their MCID.) Consequence: calibration is required at
   arbitrary thresholds, see §4.
3. **`prior_scale` range.** Resolved: LogNormal(log 0.7, 0.75) replaces Gamma(2, 1)
   (which had ~9% mass above 4).
4. **Allocation.** Decided: exact `round(N * p_alloc)` in the simulator. The
   allocation mechanism is ignorable for the posterior given the observed `group`;
   it only matters for training coverage, handled by the wider `p_alloc` range.
5. **Futility thresholds** below 0 need lower-tail calibration checks too.

## 4. Validation consequences

- SBC/tail checks draw from exactly these priors (same `prior_df`/`prior_scale`).
- Primary criterion: calibration of P(d > c) per condition, for a grid of thresholds
  c ∈ {−0.5, −0.2, 0, 0.2, 0.5} (efficacy and futility), with emphasis on the tails
  (posterior probabilities near 0.9, 0.95, 0.975, 0.99 and their complements).
- Secondary: SBC ranks/ECDF, NRMSE, contraction.
