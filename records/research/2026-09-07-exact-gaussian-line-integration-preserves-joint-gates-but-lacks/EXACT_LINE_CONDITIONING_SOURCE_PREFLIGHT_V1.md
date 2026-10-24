# Exact Gaussian line conditioning: source and variance preflight

September 7, 2026 PDT. Phase 2 only. Existing accuracy agent proposed and
checked this source; root checked the identities and the actual historical
measurement JSON/source provenance. No new network or numerical test.

## Distinct source

For deterministic unit direction a, write X=Y+T*a with independent
Y~N(0,I-aa.T), T~N(0,1). Along a fixed Y the actual finite ReLU network
is piecewise affine. If output j is alpha_kj*t+beta_kj on interval
[lo_k,hi_k], its exact conditional Gaussian mean is

    Q_j(Y) = sum_k alpha_kj*(phi(lo_k)-phi(hi_k))
                    + beta_kj*(Phi(hi_k)-Phi(lo_k)).

Then E_Y Q_j(Y)=E F_j(X). This retains actual joint gate dependence; it
is not finite-node Gauss-Hermite sampling or a Gaussian transverse closure.
The narrow Phase 2 source search found no exact whole-network breakpoint
integration implementation. That is not an exhaustive novelty claim.

## Acquisition and expected magnitude

If B_(l-1) intervals enter layer l, propagating the two coefficient rows
per interval costs 4*n^2*B_(l-1) classical leading FLOPs. Add roots,
sorting, interval splitting, coefficient assembly, Gaussian integration,
reductions, buffers and setup. Storage is O(n*max B_l). The first ReLU
generically produces1025 intervals at width1024, though specially chosen
directions can differ. No measured later interval growth or compressed
exact circuit is available. This is a cost formula with missing B_l,
not a complete numerical bill or a universal complexity bound.

Conditional one-dimensional Gaussian Poincare gives

    sum_j E_Y Var_T(F_j(Y+T*a)) <= a.T K a,
    K = E[J_F(X).T J_F(X)].

The left side is exactly the variance removed by line integration. For
an independently uniform direction, the averaged upper bound is tr(K)/n.
For any deterministic direction the upper bound is lambda_max(K).

Historical public_quadrature_derivation_v1/realnet_measurements.json has
tr(K)=922.5095376390965, lambda_max=7.686713388607837 and
sum_j Var(F_j)=68.37197541273197. Substituting these EMPIRICAL estimates
suggests removable fractions at most approximately11.24% for a best fixed
line, or1.318% on average for independent uniform directions. This does
not support the roughly99.9% variance reduction needed for sub1e-9 IID
sampling even at a one-forward-per-draw classical price. Exact line
integration costs more than one forward in the declared construction.

Critical scope: the historical receipt is observer_fit rank35, MLP89,
weights SHA d18ae98fb182d0941b0dc28905d2419ef2f1c8c281b6f904d8266fc38b9692a3.
It used160 Jacobian draws and2048 forward statistics. It is NOT current
native rank35, not a certified population spectral upper bound, and not
the variance of a high-capture controlled residual. No point estimates
are promoted to rigorous challenge-wide bounds.

## Decision

Expected actual score gain is not established. Plain exact line
conditioning is discouraged by both the variance evidence and unpriced
breakpoint growth. A different controlled residual could behave
differently, but no affordable residual with an exactly known mean and
favorable conditional variance is supplied. Missing: that concrete
residual, its actual line-conditioned variance, and complete acquisition
cost. No network run, breakpoint-rank/sampling sweep or old-family rescue
is justified. This does not exclude all conditional integration methods.

Historical measurement SHA27584cc121de0e0194f591876841475487f2382b23591bf35432ead136a9751a.
Only its summary fields and source were read; no new truth-array access.
No score change, candidate, upload, forum post, refit or frozen edit.
