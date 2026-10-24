# Fixed chi-radius fourth-plane read information test

September6,2026. Root authorized this plane-read diagnostic only.
No full carrier, wide replay, MPS, new Monte Carlo, fitting or upload.

Use all seven exact cached coherent rank35 pre-layer8..14 cells. Preserve
the current scalar trace sum(K22) at each PRE frame. Replace only the
read's supplied k4/K22/K31 by the fixed trace-shaped planes certified in
chi_radius_trace_shape_certificate_v1.json. Input mu,C,K21 remain the
original cached call arguments. New K31 is its complete physical plane,
standardized by the unchanged sigmas, not the old radial approximation.

The cached pre-C is F32-push asymmetric. Construct the new symmetric
fourth tensor from (C+C.T)/2, report that change, but retain ORIGINAL C
in both nonlinear pair reads. The target trace is the sum of the actual
original K22 array, not a differently symmetrized formula. This owns the
roundoff boundary rather than silently changing the parent covariance.

## Exact baseline, before target access

Original radial was saved as q=radial/3. Recover its unique float64 value
from the exact original k4 vector: test 3q and its immediate adjacent
doubles for byte-exact equality to radial*diag(C)*diag(C). This is finite
binary input reconstruction, not an accuracy/radius sweep. If no unique
value reproduces k4, stop; do not relax the check or replay a network.

Replay the actual Gauss6/Edge2 parent with original mu,C,K21,k4,K22. Its
mean and covariance must equal the saved actual parent arrays byte for
byte. Then provide an explicit copy of the parent's standardized K31;
that must reproduce the same mean/covariance byte-exactly. Only then use
the new full K31 and new fourth planes in the same nonlinear map.

No current-layer mean-feedback vector is applied in this established
source-v2 diagnostic: the compared pair covariance is BEFORE the
current hook. Previous coherent mean/covariance history remains fixed.
Both old and new centering, including the changed scalar mean and the
exact diagonal r2-m^2 term, are produced by the actual map. No manual
linearized covariance shift substitutes for that read.

## Fixed information gate

Only after all seven input/baseline/trace checks pass, read the original
existing8M post covariance references and fixed next-weight/response.
Require original symmetric F32 covariance targets byte-exact. Compare
the new natural-coefficient1 covariance increment against that target.

Freeze: pooled covariance-error ratio<=.95, pooled next-Gaussian-mean
response error<=.90, and at least5/7 next-value wins. Failure closes this
prescribed plane intervention, with no radius/coefficient/layer/node
sweep, fit, full-state build or candidate. A pass licenses only further
full source ownership and cost work, not automatic rollout or training.

One nice10 CPU worker, one BLAS thread, <=2GB, cached arrays only.
