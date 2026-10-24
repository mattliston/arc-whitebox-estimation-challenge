# Deterministic quadrature for m_16 at 2^41 flops: derivation and verdict

TAG quadderive-v1 | 2026-08-25 | package `public_quadrature_derivation_v1`

**Question.** Does a deterministic quadrature/cubature over the input Gaussian
(or an exact-mean control-variate hybrid) compute m_16 = E[x_16] of a
width-1024 depth-16 ReLU MLP (h_1 = W_1 z, z ~ N(0, I_1024),
x_L = relu(h_L), h_{L+1} = W_{L+1} x_L, weights exact) to 1e-9-class MSE
inside B = 2^41 metered flops?

**Verdict: NO — infeasible, with a measured impossibility bound.** The best
sampling-class rule at budget is the Gaussian-equivalent affine control
variate at ~0.89x plain MC; every deterministic rule tested or bounded is
worse than plain MC at matched counts; 1e-9 needs a surrogate/rule capture of
99.9% and the measured ceiling of every exactly-integrable family is <= 43%
(degree <= 2 oracle) / <= 63% (perfect 512-dim-subspace oracle). Numbers
below; assembled in `QUADRATURE_VERDICT.json`.

Budget frame used throughout (mult+add counted separately, per the board
convention): forward = 2·16·1024² = 2^25 flops, so N_plain = 2^41/2^25 =
65,536 forwards; the CV hybrid (precompose 3.22e10 + 2048 calibration
forwards + affine eval 2.1e6/sample) leaves N_pairs = 58,845 corrected
samples.

---

## 1. Task 1 — what smoothness is actually available (and what is not)

**The "each layer's integral smooths the next layer's kinks" premise is false
for this architecture.** There is no fresh noise between layers: the network
is a single deterministic, continuous, piecewise-linear map f = x_16(z), and
m_16 is ONE Gaussian integral of it over R^1024. Layered smoothing exists
only behind inner conditional expectations (Fubini: F(z_A) = E_{z_B}[f] is
C-infinity in z_A because kink locations move transversally with z_B), but F
is itself an intractable integral — any estimator only evaluates f at points,
so the smoothness of F is purchasable only through further sampling, which
reinstates the 1/N variance floor. Exact statement:

- f is globally Lipschitz, piecewise linear on O(16·1024) curved kink cells
  (layer-1 kinks are the 1024 hyperplanes {w_i·z = 0}; deeper kinks are
  curved). Its second distributional derivative is a surface measure: f lies
  in H^{3/2-eps} and in NO mixed-smoothness class W^{2,mix}. Consequences:
  - tensor/sparse-grid (Smolyak) fast rates need dominating mixed
    smoothness — absent;
  - Gauss-Hermite per-axis spectral rates need per-axis analyticity — absent
    beyond layer 1 (kink normals become dense in direction space);
  - class-level curse: worst-case deterministic error over an H^{3/2} ball
    in d = 1024 dims scales as N^{-3/2/1024} = N^{-0.0015} — flat.
- The ONLY exploitable structure is spectral concentration: (i) low
  Hermite-degree mass (polynomial-exact rules, QMC, polynomial surrogates),
  (ii) low-dimensional (active-subspace) mass (GH-on-top-k x MC, ridge
  surrogates). Both were MEASURED on the real net (Sections 3-4); both cap
  far below what 1e-9 requires (Section 6).

## 2. The annealed degree law (theory), and how the real nets break it

Mehler's identity: for any L2 function, E[f(z) f(z')] with corr-rho Gaussian
pairs equals sum_k M_k rho^k where M_k is the variance fraction at Hermite
degree k. So the centered cross-input correlation curve c(rho) IS the degree
spectrum, and it is measurable with forward passes alone.

For one ReLU layer at exactly-Gaussian preactivations, c is the centered
arccos kernel c(rho) = [sqrt(1-rho^2) + rho(pi - arccos rho) - 1]/(pi-1) with
exact coefficients m_1 = (1/4)/sigma^2 = 0.733471, m_2 = 0.233471,
m_4 = 0.019456, ... (sigma^2 = 1/2 - 1/(2pi); closed forms in
`theory_branching.py`). c is a probability generating function, so depth-D
composition is a Galton-Watson branching process: the Hermite degree of the
depth-D net is generation-D total progeny with offspring mean
mu = c'(1) = pi/(pi-1) = 1.46694. Exact windowed-convolution table
(`theory_branching.json`):

| depth | M(deg=1) | M(<=2) | M(<=8) | M(<=64) | mean deg | median |
|---|---|---|---|---|---|---|
| 6  (annealed) | 0.1557 | 0.3127 | 0.7549 | 0.9841 | 10.0 | 4 |
| 16 (annealed) | 0.00702 | 0.0153 | 0.0635 | 0.3465 | 460 | 120 |

**Measured on the real net (rank 35), the annealed law holds exactly where it
must and breaks where fixed weights matter.** Layer-1 linear mass measured
0.734 vs 0.733471 predicted (exact: layer-1 preactivations are exactly
Gaussian). Deeper, the per-layer linear-mass retention drifts up from the
annealed 0.7335 toward ~1 and the linear mass SATURATES at ~0.22 instead of
decaying to 0.7%:

layer:    1     2     3     4     5     6     8     10    12    14    16
linear:  0.734 0.596 0.508 0.446 0.400 0.359 0.311 0.269 0.247 0.231 0.222

The weights themselves are unremarkable (per-layer rms singular gain 1.41 =
sqrt(2), He-critical, smax ~ 2.82): the deviation is finite-width coherence —
the product of specific fixed matrices preserves linear signal that the
phase-averaged (annealed, width->infinity) map destroys. Section 7's n=64
ensemble shows the same excess in the ENSEMBLE MEAN with small per-net
scatter: it is a systematic finite-width correction, suppressed with width
far slower than 1/n, not a peculiarity of the ARC generator. Three independent measurements of the final
linear (Hermite-1) mass agree: Richardson curve slope 0.222 +- 0.04, debiased
Stein ||E[J]||_F^2/trVar = 0.210, paired affine-surrogate capture 0.205
(optimal per-coord beta 0.206).

Full measured degree mass of the real net (centered curve + NNLS,
rho=0 null passes at 0.005 +- 0.007; `curve_v2.json`, `nnls_v3.json`):

| M(<=1) | M(<=2) | M(<=4) | M(<=8) | M(<=16) | M(<=64) | M(>64) |
|---|---|---|---|---|---|---|
| 0.224 | 0.432 | 0.480 | 0.765 | 0.845 | 0.972 | 0.028 |

So the real nets are FAR more low-degree than annealed theory predicts
(median degree ~3-5 vs 120) — deterministic structure is genuinely present —
and yet (Section 6) still nowhere near enough for 1e-9.

## 3. Active subspace (truncation dimension), measured

C = E[J^T J] over 160 exact Jacobians (J = prod D_l W_l^T), eigendecomposed
(`realnet_measurements.json` p2): trace fractions need k = 132 directions for
50%, 455 for 90%, 751 for 99%, 896 for 99.9%. The gradient spectrum is NOT
spiky: no few-direction representation exists. Poincare slack
trC/trVar = 13.5 (gradient energy >> variance: high-degree content).

The operationally decisive number is the conditional-variance floor
E[Var(f | top-k subspace)] — the exact variance floor of ANY rule that
integrates the top-k subspace PERFECTLY and samples the complement (nested
MC, p3):

| k | 8 | 32 | 128 | 512 |
|---|---|---|---|---|
| capture | 6.5% | 10.2% | 27.7% | 63.1% |

A GH-tensor on k dims costs an m^k-point grid per complement sample. It beats
plain MC only if capture_k > 1 - 1/m^k; measured capture at k = 2 is 1.8-3%
vs the required 98.4% (m = 8). Measured on the real net: the GH2 hybrid is
~60x WORSE than MC at matched forwards (Section 5). Larger k only worsens
the grid factor (2^k growth vs capture growth from 6.5% to 63%).

## 4. Task 3 — the exact-mean control-variate hybrid: exact statement and cap

**Cancellation (already on record in `cp1792_rqmc_nonlinear_residual`):**
closure + Q_N[f - const] == Q_N[f]; a constant control does nothing. An
unbiased hybrid needs a PATHWISE surrogate s(z) with exactly computable
Gaussian mean: est = E[s] + (1/N) sum [f(z_i) - s(z_i)], which is unbiased
for ANY s (calibration quality affects only variance), with
MSE = Var[f - s]/N = vbar·(1 - capture)/N_pairs.

**The Gaussian-equivalent affine network exists and is exactly integrable.**
With per-layer preactivation marginals (mu_l, s_l) (2048 calibration
forwards), define a_l = Phi(mu_l/s_l) (the Stein-optimal linearization slope:
a* = Cov(relu(h), h)/Var h = E[relu'(h)] = Phi(mu/s) for h ~ N(mu, s^2)) and
m_l = mu_l Phi(mu_l/s_l) + s_l phi(mu_l/s_l) (the exact ReLU Gaussian mean).
The surrogate chain
    x~_l = a_l ∘ (W_l x~_{l-1} - mu_l) + m_l
is affine in z, precomposes to s(z) = A_eff z + c_eff (3.22e10 flops once,
2.1e6 flops/eval), and its exact Gaussian mean is c_eff — the chain of the
closure-style per-layer Gaussian means. If the marginals are taken from the
closure instead of calibration forwards, E[s] IS the closure's emitted mean
chain: the task's proposed statement holds exactly.

**But its capture is algebraically capped by the linear mass:** s is affine,
so corr^2(f, s) <= M(<=1) = 0.21-0.22. Measured capture 0.205 (paired), and
per-coordinate optimal rescaling adds nothing (0.206): the Stein chain is
already at the linear ceiling. Two exact side facts, both verified:
- affine CV + antithetic == antithetic alone (an affine surrogate's centered
  part is odd; pair sums cancel it identically);
- antithetic VR = 1 + c(-1) = M_even - M_odd + 1: measured c(-0.95) = -0.102
  so VR_anti ~ 0.89 — antithetic barely helps BECAUSE the even mass
  correlates pairs positively (this is the mechanistic explanation of the
  standing "antithetic/QMC don't help" board observations).

Richer exactly-integrable surrogates and their measured ceilings:
- any polynomial surrogate of Hermite degree <= 2 (the largest class whose
  Gaussian mean is exactly computable at O(d^2) cost): capture <= M(<=2) =
  0.432 — an ORACLE bound; actually fitting Gamma costs forwards and fit
  noise only lowers it;
- ridge families on k directions (means exactly computable by k-dim
  quadrature): capture <= conditional capture at k: 27.7% at k = 128, 63.1%
  at k = 512 (oracle; the 512-dim ridge function is itself unlearnable and
  unintegrable in practice);
- a per-input pathwise surrogate whose exact mean equals the CLOSURE's
  3.3e-8 output would need corr(f, s) >= 99.95%: no such object exists in
  the tractable algebra — this is the same wall as the program's
  pushforward/observer results (the closure is a moment recursion with no
  pathwise form), now with a measured quantitative gap (0.21 vs 0.9995).

## 5. Measured rules race (real net, rank 35, vs sealed truth)

`realnet_rules.json`; per-coord MSE, mean over reps (se ~ 10-40%):

| rule | N forwards | MSE | vs plain MC at same N |
|---|---|---|---|
| MC | 1024 | 6.33e-5 | 1.00 |
| CV-affine | 1024 | 5.39e-5 | 0.852 |
| MC | 4096 | 1.89e-5 | 1.00 |
| CV-affine | 4096 | 1.44e-5 | 0.761 |
| ANTI | 4096 | 1.58e-5 | 0.836 |
| GH2 (8^2 grid x MC) | 4096 | 1.011e-3 +- 3.8e-5 | **53.4x WORSE** |
| Lattice (Korobov a=1465, 8 shifts) | 4093 | 1.41e-5 +- 1.1e-6 | 0.744 (1.16x RMSE) |
| MC | 16384 | 3.70e-6 | 1.00 |
| CV-affine | 16384 | 3.12e-6 | 0.844 |

MC matches vbar/N at every N (vbar = 6.68e-2 on this row); fitted MSE
exponents: MC -1.025, CV -1.028 — the 1/N law, no rule changes the exponent.
GH2 measured 53.4x worse vs ~62x predicted from grid-factor x (1 - capture_2)
= 64 x 0.982. The lattice's 1.34x variance gain is exactly the partial
integration of the low-degree mass and is the LARGEST deterministic-flavored
gain observed anywhere in this study.

Cross-row robustness (ranks 36/120/234 vs 35): vbar = 6.43e-2 / 8.62e-2 /
8.17e-2 (vs 6.68e-2) — same scale on all four rows; affine capture 0.196 /
0.190 / 0.208 (vs 0.205); linear mass 0.231 / 0.282 / 0.195 (vs 0.222). Every
verdict-carrying fraction is row-robust. NOTE: the briefing figure "plain MC
~1.5e-7 at 65k" does not match plain MC on any measured row (vbar/65536 =
1.0-1.3e-6); that figure evidently describes the shipped MC-HYBRID card
(closure + variance-reduction stack), not plain MC. All arithmetic here uses
the directly measured vbar.

## 6. The impossibility arithmetic

All numbers row-35 scale (vbar = 6.68e-2, and the three cross-check rows sit
at 6.4-8.6e-2, so this scale is representative of the fleet; fractions and
captures are row-robust per Section 5).

Required for MSE T with an unbiased hybrid at N_pairs = 58,845:
capture >= 1 - T·N_pairs/vbar:

| target T | required capture |
|---|---|
| 1e-9 | **99.912%** |
| 3.3e-8 (match closure, row scale) | 97.09% |
| 1.019e-6 (match plain MC) | 10.2% |

Measured/oracle ceilings vs that requirement:

| family | capture ceiling | MSE at budget |
|---|---|---|
| affine (Stein chain), measured | 0.205 | 9.02e-7 |
| degree <= 2 polynomial surrogate, oracle | 0.432 | 6.45e-7 |
| degree <= 3 exact rule (Stroud 2d pts, scrambled, 32 repeats) | 0.480 | 5.30e-7 |
| 128-dim perfect ridge/subspace, oracle | 0.277 | 7.37e-7 |
| 512-dim perfect subspace, oracle | 0.631 | 3.76e-7 |
| degree <= 16 exact rule, oracle (no implementation exists at d=1024) | 0.845 | 1.58e-7 |
| degree <= 64 exact rule, oracle | 0.972 | 2.85e-8 |
| required for 1e-9 | 0.99912 | 1e-9 |

Structural closures of the implementable families:
- polynomial-exactness: degree-3 Gaussian rules cost 2d = 2048 points
  (fits), degree-5 already costs ~2d^2 = 2.1e6 points = 32x the entire
  budget (Stroud / Smolyak level-2), degree-7+ grows another factor ~d;
  the capture ladder above shows even FREE degree-64 exactness only reaches
  closure-class 2.85e-8, still 28x above 1e-9 — the measured 2.8% mass above
  degree 64 is by itself a floor at this budget;
- QMC/lattice: gains are confined to low-degree/low-superposition mass and
  are constant-factor (lattice exponent -1.09): measured 1.34x variance
  (1.16x RMSE) at n=1024 with a 4093-point Korobov rule (best of 48
  generators), 2.6x variance at n=64 — shrinking with dimension, consistent
  with the board-measured QMC null at full budget;
- GH x MC hybrids: grid factor m^k always exceeds 1/(1-capture_k) by
  orders (measured 60x loss at k=2; capture reaches only 63% at k=512 where
  the grid is astronomically impossible);
- purely deterministic single-shot rules: no variance, but instance bias on
  the >= 97% uncaptured mass cannot be certified below the class bound
  (H^{3/2}, N^{-0.0015}) — and every implementable instance measured LOSES
  to plain MC at matched N.

**Conclusion: at 2^41 flops the entire quadrature/cubature/CV class lands in
the 4e-7..1e-6 band on this row (fleet-scale: ~0.6-1.5e-7), i.e. at or above
plain MC's 1.0e-6 (row) / 1.5e-7 (fleet) with at best a 1.13x improvement —
while the analytic closure already sits at 3.3e-8 (fleet). 1e-9 is
unreachable by this class by ~3 orders of magnitude in variance.** The gap
to 1e-9 is not a rule-selection problem: it is the 2.8-15% Hermite mass above
degree 16-64 spread over ~750 active directions, which no exactly-integrable
structure reaches. The only doors this derivation leaves open are the ones
the program is already pursuing: making the analytic carry itself more exact
(closure supply), or finding a pathwise-coupled surrogate OUTSIDE the
exactly-integrable algebra — i.e., not quadrature.

## 7. Task 4 — small-n verification (n=64, depth 6) and fail-fast

`quad_verify.json`. E1, ensemble annealed law (24 random He nets, per-layer
linear mass, ensemble mean +- sd vs the branching prediction):

| layer | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| measured mean | 0.733 | 0.601 | 0.522 | 0.464 | 0.423 | 0.397 |
| annealed | 0.733 | 0.538 | 0.395 | 0.289 | 0.212 | 0.156 |
| ensemble sd | 0.006 | 0.013 | 0.018 | 0.017 | 0.025 | 0.033 |

Layer 1 agrees exactly (it must: exactly-Gaussian preactivations). Deeper,
the EXCESS low-degree mass over annealed is systematic (present in the
ensemble mean, small scatter): finite-width coherence, suppressed with width
but much slower than 1/n (layer-6 linear mass: 0.397 at n=64, 0.359 at
n=1024-real, vs 0.156 annealed). The annealed branching law is the exact
width->infinity limit and the correct lower envelope; real nets carry MORE
low-degree mass — which is why every capture ceiling in Section 6 was
MEASURED on the real net rather than taken from theory.

E2, estimator race on one fixed net vs a 2^24-antithetic truth (truth floor
1.2e-8, 3 orders below the smallest measured MSE; 24 reps; predictions
written into the JSON before the race):

- predicted VR_CV = 0.621 (direct paired variance); measured CV/MC ratio
  across N = 2^7..2^14: 0.59-0.67, mean ~0.62. Prediction = measurement.
- predicted VR_ANTI = 1 + c(-1) = 0.807; measured 0.71-1.02 (noisy,
  consistent).
- GH2 hybrid: 33x worse than MC (grid factor 64 x (1 - capture_2), as
  derived).
- Lattice: best non-MC result anywhere: 2.6x variance = 1.62x RMSE gain at
  N = 4093 in d = 64 — and it decays with dimension (only 1.16x RMSE at the
  real d = 1024).
- fitted MSE exponents: MC -0.980, CV -0.989, ANTI -0.999, GH2 -0.948,
  LAT -1.087. **Every rule sits on the 1/N line; no rule changes the
  exponent.** Gains are constant-factor only.

**FAIL-FAST (preregistered): does any rule beat sqrt-N MC by > 3x RMSE at
matched counts? NO — best 1.62x (n=64, quadrature's EASIEST regime), 1.16x
at the real geometry. The criterion triggers: impossibility bound written,
route closed.**

## 8. Receipts and discipline

- Orientation/convention: layer-1 analytic mean vs sealed truth rel-RMS
  4.7e-5 (f32 rounding scale) under x <- relu(x @ W); the row-convention
  alternative mis-fits at 3.2e-2 — convention pinned.
- MC scaling: 2048-forward mean vs truth MSE 3.74e-5 vs predicted 3.26e-5.
- Curve estimator null at rho=0: 0.005 +- 0.007 (v2, centered); the v1
  uncentered estimator FAILED this null (+0.147) and was superseded — kept
  on disk as the incident record.
- Truth access: only the lawful observer_fit parquet row(s); no fits, no
  submissions, no frozen-package edits; single-process one-forward-at-a-time
  at n=1024; peak RSS 0.9 GB.
