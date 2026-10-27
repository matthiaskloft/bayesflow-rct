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
N        ~ log-uniform-int(20, 1000)       # optional 70/30 tilted mixture: see §2
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
| `prior_df` | condition | 1 + LogNormal(log 5, 1.0) | log-scale, shifted so df ≥ 1 (no undefined-mean priors; Cauchy df = 1 is the boundary): 95% 1.7–36.5, P(df < 3) ≈ 18% (covers rctbayespower default df = 3), P(df < 2, infinite variance) ≈ 5%, P(df > 30) ≈ 4% (≈ Normal, no separate Normal switch). Chosen over Gamma(2, 0.1) (Juárez & Steel, 2010, working-paper Eq. 11; also the brms default for Student-t df, Bürkner, 2017), which has only ~4% below df = 3. That Gamma prior was proposed for the df of the data distribution in a panel model, not for a prior on an effect, so it is a reference point only. Finite-df t over Gaussian and Cauchy (t1) as heavy-tailed default: Gelman et al. (2008); their scale 2.5 is a logit-scale value and does not transfer to d. See `references.md`. | review |
| `prior_scale` | condition | LogNormal(log 0.7, 0.75) | median at the BayesFactor default r = √2/2 ≈ 0.707 for a Cauchy prior on d (Morey & Rouder, 2018; see Tendeiro & Kiers, 2019). Correction: Rouder et al. (2009) propose the Cauchy prior on effect size but use r = 1 as the default (p. 233). 95%: 0.16–3.0 covers sceptical to ultrawide (√2); P(>4) ≈ 1%. | review |
| `R2` (covariate) | nuisance | Beta(1.2, 2) | right-skewed (mode 0.17, mean 0.37), broad: 5–95% 0.04–0.80, P(R² > 0.9) ≈ 1.3%. Chosen for now; revisit after first training run. With N ≥ 20 the data identify R², so prior influence on the marginal posterior of d is small. (TBD) no empirical distribution of R² found. Range is plausible: simulated baseline–outcome correlations 0.1–0.9, i.e. R² 0.01–0.81, are argued typical by Egbewale et al. (2014); one observed trial outcome had r = 0.44 (R² ≈ 0.19; Kahan et al., 2014); R² > 0.3–0.5 is what makes covariate adjustment clearly pay off (Siegfried et al., 2023). See `references.md`. | review |
| intercept, σ | — | removed by standardization | d is invariant to affine transforms of y | — |
| `N` | condition | log-uniform on 20–1000 (`n_log_share = n_log_alpha = 1`). Shares: 10% at N < 30, 18% at N ≥ 500. A 70/30 mixture of a small-N-tilted log-uniform (alpha 0.7) and a uniform, adapted from the bayesflow-irt "mixed" size sampler (EXP-0016, PR #159), was tested in v3 and was not better calibrated across two seeds; seed-to-seed variation at N = 1000 exceeded the sampler effect, and 40k training steps (vs 20k) mattered more | training design, not a prior on data | v2; mixture tested in v3 |
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

## 5. Sufficient-statistic conditions (late fusion)

`SufficientStatistics` (adapter) adds exact sufficient statistics of the normal ANCOVA
as extra `inference_conditions`.

- **Why they are enough.** Given the design, (X'X, X'y, y'y) are sufficient for (β, σ²);
  the posterior depends on y only through them (factorization theorem; Robert, 2007,
  Section 1.3; Gelman et al., 2013, Section 14.2, where β̂ = (X'X)⁻¹X'y and s² carry the
  posterior). Treating x as fixed loses nothing because the covariate distribution carries
  no parameters of (β, σ²): the posterior factors as p(ψ|X)·p(θ|X,y) under prior
  independence (Gelman et al., 2013, Section 14.1, p. 354).
- **Reduction.** X = [1, g, x]. After per-dataset standardization Σx = Σy = 0 and
  Σx² = Σy² = N − 1; n₁ = Σg and N are already conditions. The remaining unique entries
  of X'X and X'y are Σgx, Σgy and Σxy, i.e. x̄₁, ȳ₁ and r_xy (= Σxy/(N − 1)). Three
  numbers, whatever N.
- **Easier encodings.** d̂ = β̂_g/σ̂ (σ̂² = RSS/(N − 3)) and log se(d̂) are exact functions
  of these three plus (n₁, N), added only because they are close to the target scale.
  se(d̂) = √([(X'X)⁻¹]_gg + d̂²/(2(N − 3))) is the usual large-sample variance of a
  standardized mean difference, 1/n₁ + 1/n₂ + d²/(2(n₁ + n₂)) in the two-sample case
  (Hedges & Olkin, 1985; Borenstein et al., 2009), with [(X'X)⁻¹]_gg in place of
  1/n₁ + 1/n₂ and the residual df N − 3 in place of n₁ + n₂ (the two-sample formulas
  use the total N; a df of N − 3 is the regression analogue). Not checked against the
  full text of either book (see `references.md`).
- **Fusion.** The features go straight into `inference_conditions` (late fusion). The
  DeepSet (summary_dim 4) is kept as a robustness path, e.g. if the model is misspecified.
- **Numerical check.** d̂ equals the raw-data OLS value exactly; (d̂ − d)/se(d̂) has
  SD ≈ 1.01 at N = 20 and N = 200 over 4,000 simulations (SD 1 expected).
