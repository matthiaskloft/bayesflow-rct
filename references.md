# References

APA 7. Verified against full text (Zotero index or open-access PDF/XML) and OpenAlex
metadata, October 2026. Each entry ends with the claim it supports in
`docs/ancova_prior_spec.md`.

## Prior on the effect size `d` (`prior_scale`, `prior_df`)

Bürkner, P.-C. (2017). brms: An R package for Bayesian multilevel models using Stan.
*Journal of Statistical Software, 80*(1), 1-28. https://doi.org/10.18637/jss.v080.i01

> Supports: `prior_df` row. Shows that the Gamma(2, 0.1) prior on Student-t df is the
> brms default, attributed to Juárez and Steel (2010) (Section 2, model description: "By
> default, nu has a wide Gamma prior").

Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008). A weakly informative default
prior distribution for logistic and other regression models. *The Annals of Applied
Statistics, 2*(4), 1360-1383. https://doi.org/10.1214/08-AOAS191

> Supports: `prior_df` row (qualitatively). Section 2 (prior specification) recommends Student-t priors, with
> the Cauchy (t1) scale 2.5 as default, on inputs scaled to SD 0.5, and finds a finite-df
> t beats the Gaussian on average (Section 5, Figure 6). The 2.5 is a logit-scale value for
> logistic regression and does not transfer to Cohen's d; the paper's own empirical
> scale for logistic coefficients is 0.75 (discussion of the scale choice).

Juárez, M. A., & Steel, M. F. J. (2010). Model-based clustering of non-Gaussian panel
data based on skew-t distributions. *Journal of Business & Economic Statistics, 28*(1),
52-66. https://doi.org/10.1198/jbes.2009.07145

> Supports: `prior_df` row, the Gamma(2, 0.1) prior on df (shape 2, rate 0.1; mean 20,
> variance 200). Checked in the open working-paper version (Juárez & Steel, 2006,
> CRiSM Paper 06-14, Section 2.1, Eq. 11: p(nu) proportional to nu exp(-nu/10)). The
> published JBES text is paywalled and was not read. The prior is for the df of the
> data distribution in a panel model, not for the df of a prior on an effect.

Morey, R. D., & Rouder, J. N. (2018). *BayesFactor: Computation of Bayes factors for
common designs* (R package). https://CRAN.R-project.org/package=BayesFactor

> Supports: `prior_scale` row, the sqrt(2)/2 default. Not read directly. The sqrt(2)/2
> default ("medium") for `ttestBF` is documented by Tendeiro and Kiers (2019, Point 3) and
> Schönbrodt et al. (2017, Section "Selecting an effect size prior"), which cite this
> package. Check the package documentation before relying on the exact argument name.

Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D., & Iverson, G. (2009). Bayesian t
tests for accepting and rejecting the null hypothesis. *Psychonomic Bulletin & Review,
16*(2), 225-237. https://doi.org/10.3758/PBR.16.2.225

> Supports: Cauchy prior on the effect size delta (JZS prior section; delta ~ Cauchy). The scale is
> generalised to r times Cauchy on p. 233, where "the scale r is implicitly set to 1.0"
> and "r = 1.0 is recommended". The paper does not use sqrt(2)/2.

Schönbrodt, F. D., Wagenmakers, E.-J., Zehetleitner, M., & Perugini, M. (2017).
Sequential hypothesis testing with Bayes factors: Efficiently testing mean differences.
*Psychological Methods, 22*(2), 322-339. https://doi.org/10.1037/met0000061

> Supports: `prior_scale` row, secondary source. Notes the BayesFactor package offers
> r = sqrt(2)/2, 1 and sqrt(2) (p. 326).

Tendeiro, J. N., & Kiers, H. A. L. (2019). A review of issues about null hypothesis
Bayesian testing. *Psychological Methods, 24*(6), 774-795.
https://doi.org/10.1037/met0000221

> Supports: `prior_scale` row, provenance of sqrt(2)/2 (p. 781, Point 3): "The scale
> parameter has been fixed at one (Rouder et al., 2009) and later on at sqrt(2)/2 = .707
> (e.g., Morey & Rouder, 2018)". Also: Cauchy scale r puts 50% of prior mass in (-r, r).

## Covariate-outcome association (`R2`)

Egbewale, B. E., Lewis, M., & Sim, J. (2014). Bias, precision and statistical power of
analysis of covariance in the analysis of randomized trials with baseline imbalance: A
simulation study. *BMC Medical Research Methodology, 14*, Article 49.
https://doi.org/10.1186/1471-2288-14-49

> Supports: `R2` row, range only. Simulated baseline-outcome correlations 0, 0.1-0.9
> (R2 = 0.01-0.81), justified as the range "between baseline covariates and outcome scores
> in RCTs may range" (Methods, levels of experimental conditions). This is an argued
> range, not an empirical distribution. Also reports SE ratio ANCOVA/ANOVA close to
> sqrt(1 - rho^2) (Results, Precision).

Kahan, B. C., Jairath, V., Doré, C. J., & Morris, T. P. (2014). The risks and rewards of
covariate adjustment in randomized trials: An assessment of 12 outcomes from 8 studies.
*Trials, 15*, Article 139. https://doi.org/10.1186/1745-6215-15-139

> Supports: `R2` row, one data point. In the MIST2 trial the continuous outcome's
> baseline correlation was 0.44 (R2 about 0.19) (Methods, increase in power for
> continuous outcomes). Correlations for the other outcomes are not tabulated.

Siegfried, S., Senn, S., & Hothorn, T. (2023). On the relevance of prognostic information
for clinical trials: A theoretical quantification. *Biometrical Journal*, Article
e2100349. https://doi.org/10.1002/bimj.202100349

> Supports: `R2` row, plausibility of the upper-middle range. One worked case study had
> out-of-sample explained variability of at most 36% (Section 3.4); the paper treats
> R2 > 0.3-0.5 as what is needed for a worthwhile (>20%) sample-size reduction
> (Sections 3.1, 5). Read via an automated summary of the Europe PMC full text; confirm
> the quotes in the PDF before citing numbers.

## Sufficient-statistic conditions (`SufficientStatistics`)

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B.
(2013). *Bayesian data analysis* (3rd ed.). CRC Press.
https://doi.org/10.1201/b16018 (ISBN 978-1-4398-4095-5)

> Supports: ancillarity of X. Section 14.1, "Formal Bayesian justification of conditional
> modeling" (p. 354): with independent priors on psi and theta, "the posterior
> distribution factors" and p(theta|X,y) can be analysed alone "with no loss of
> information". Section 14.2 (p. 355): posterior of beta given sigma is N(beta-hat,
> V sigma^2), beta-hat = (X'X)^-1 X'y; sigma^2|y is scaled Inv-chi^2(n-k, s^2). Read in
> the Zotero full text (author's PDF, errors fixed 13 Feb 2020). DOI per OpenAlex.

Robert, C. P. (2007). Introduction. In *The Bayesian choice* (2nd ed., pp. 1-49).
Springer. https://doi.org/10.1007/0-387-71599-1_1

> Supports: sufficiency and factorization. Section 1.3: a statistic is sufficient if the
> distribution of x given T(x) "does not depend on theta"; the factorization theorem gives
> sufficient statistics (worked normal example); the posterior "depends on x only through"
> the likelihood. Exercise 1.22 states x = t(z) is sufficient iff pi(theta|x) = pi(theta|z)
> for every prior. Read in the Zotero full text.

## Not obtained

- Hedges, L. V., & Olkin, I. (1985). *Statistical methods for meta-analysis*. Academic
  Press. https://doi.org/10.1016/C2009-0-03396-0 (ISBN 978-0-12-336380-0). Metadata
  confirmed in OpenAlex; full text not available, so the variance of d formula is unchecked.
- Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2009).
  *Introduction to meta-analysis*. Wiley. https://doi.org/10.1002/9780470743386 (ISBN
  978-0-470-05724-7). Metadata confirmed in OpenAlex; full text not available (publisher
  PDF returned 403), so the d variance formula is unchecked.
- Borm, G. F., Fransen, J., & Lemmens, W. A. J. G. (2007). A simple sample size formula
  for analysis of covariance in randomized clinical trials. *Journal of Clinical
  Epidemiology, 60*(12), 1234-1238. https://doi.org/10.1016/j.jclinepi.2007.02.006
  (closed access; full text not available, so no numbers verified).
- Bloom, H. S., Richburg-Hayes, L., & Black, A. R. (2007). Using covariates to improve
  precision for studies that randomize schools to evaluate educational interventions.
  *Educational Evaluation and Policy Analysis, 29*(1), 30-59.
  https://doi.org/10.3102/0162373707299550 (closed access; not read). Hedges and Hedberg
  tables were likewise not obtained.
