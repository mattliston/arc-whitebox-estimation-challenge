# Last-two-layer precision: what the born error is, and why no local computation removes it (fable_physicist_v1/late, 2026-09-22 night)

Torch card (model_bank_oldtier.py; unmodified path bit-exact with the bank port), learned parts ON, no refit. Truths: 1e9
truthpost means, independent 32M per-unit cumulants, 8M post covariances and planes. Ratios are final MSE relative to the base card.

## 1. The error born at L15 is input error; the final map is exact (`late_oracle.py`)

Error born at L15 given an exact post mean at L14 (m@14):

| added at L15, on top of m@14 | r35 | r100 |
|---|---|---|
| nothing (born at L15) | 0.232 | 0.232 |
| exact per-unit v | 0.143 | 0.121 |
| exact k3 | 0.205 | 0.210 |
| exact k4 | 0.214 | 0.221 |
| exact v + k3 + k4 | 0.077 | 0.072 |
| exact v + k3 + k4, learned mean correction off | **0.013** | **0.009** (MC noise floor) |
| learned mean correction off, card inputs | 0.281 | - |
| exact full L14 post covariance (8M) | - | 0.133 |
| exact off-diagonal L14 post covariance only | - | 0.136 |

nonlinear_scalar with exact inputs and no learned correction lands at the noise floor. The whole 23% born at L15 is per-unit input
error: variance first (38-48%), then k3 (10-11%) and k4 (5-8%). The learned correction repairs part of it with card inputs and hurts
with exact ones.

Error born at L14 given an exact L13 mean (m@13; ratio 0.415 r35 / 0.414 r100):

| added at L14, on top of m@13 | r35 | r100 |
|---|---|---|
| exact v | 0.346 | 0.342 |
| exact k3 | 0.414 | - |
| exact k4 | 0.382 | 0.398 |
| exact v + k3 + k4 | 0.311 | 0.294 |
| exact off-diagonal L13 post covariance | 0.336 | 0.293 |
| truth K22 + K31 planes at L14 | 0.392 | 0.395 |
| truth K21 plane at L14 | 1.50 | - |
| learned mean correction off at L14 | 0.462 | - |

## 2. The variance error is carried off-diagonal covariance, orthogonal to the mean (`v15_error_anatomy.py`, r35)

v_15,j = w_j^T C_14 w_j. Push the card-minus-truth L14 post covariance through W_15 and it reproduces the card's v_15 error
(cosine 0.96). Weighted by the mean sensitivity pdf/(2 sigma), the per-unit post-mean error it causes (rms) comes from:

| part of the L14 covariance error | mean error rms |
|---|---|
| total | 4.9e-5 |
| off-diagonal part | 4.9e-5 |
| per-unit post variances (diagonal) | 0.4e-5 |
| pair-map output | 4.9e-5 |
| learned covariance corrections | 0.9e-5 |
| mean-orthogonal part | 4.7e-5 |
| part along the mean | 1.3e-5 |

L13 into v_14 is the same. The pair map is exact on true inputs, so this is carried error: it accumulates from each layer's plane
reads. The lead's cov-only oracle shows the three planes together explain it, and K21 alone does not. No computation local to L14/L15
sees it without new information.

## 3. Every local candidate, measured

| candidate | result | verdict |
|---|---|---|
| zero-cost per-unit k4 at L13-15 (fit 0.5 radial v^2 + 0.9 k3^2/v; centered R2 0.61 -> 0.80-0.90 unweighted) | closed loop +21% (L15), +15% (L14), +40% (both), +61% (L13-15) on all four nets | dead |
| why: kink units | truth k4 is about radial v^2 there (median ratio 0.96-1.00); the fit halves it and doubles the mean-weighted k4 error (3.7e-5 -> 7.2e-5) | - |
| sensitivity-weighted k4 structures (v^2 + k3^2/v, mu k3, alpha^2 v^2) | best cuts the weighted error 3-5%; noise is 15x smaller than the error | dead |
| index-class feeds for k4_15 from truth L14 post planes (W^4 k4, (2,2), (3,1)) | R2 +0.00 / +0.00 / +0.05 | dead |
| exact untruncated old tier (no frame at all) | +1.7% (r35), -4.0% (r100); L15 k3 error unchanged (6.39% -> 6.35%) | not a driver |
| rescaling the L15 k3 carriers (old, young, star diagonal, b) | best coefficients 0.97-1.06; weighted error -2-3% | dead |
| higher-order pair series | converged (lead's series_order_test: <= 0.08%) | dead |
| better final map | exact with exact inputs (section 1) | nothing to gain |

## 4. Answer

The born mean error at L14 and L15 is per-unit input error of the kink units. The largest part is variance, and that is the carried
mean-orthogonal off-diagonal covariance of the previous layer. Its only fix is upstream: the three planes' covariance read at every
layer (the lead's and covhunter's lane). The per-unit k3 and k4 errors at the last layers are unit-specific. The card's isotropic k4 is
already right on average for kink units. Every cheap structure is at its floor, so a local fix needs a truth-grade per-unit supply.
None of the listed last-layer computations removes the born error at +3-6% of the bill.
