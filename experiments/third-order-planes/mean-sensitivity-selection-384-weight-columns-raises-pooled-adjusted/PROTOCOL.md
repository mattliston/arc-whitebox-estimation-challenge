# Fixed local mean-sensitivity Omega selection on submitted Lazy

Prospective Phase2 protocol, 2026-09-19 PDT. Parent is submitted Lazy #331599,
source b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c,
official adjusted score 3.8164492657795035e-9. No outputs or accuracy labels
for this candidate have been evaluated. The approximately1e-9 goal is active.

At joins5..14 replace the parent's first384 current-weight columns with the
384 columns having greatest local Gaussian mean sensitivity to third
cumulant. With the already computed preactivation mean mu and variance vv:

    v = maximum(vv, 1e-10)
    alpha = mu / sqrt(v)
    score = abs(alpha) * normal_pdf(alpha) / v
    indices = stable_argsort(-score)[:384]
    Omega = w32[previous[:,None], indices[None,:]]

The leading cumulant expansion gives d E[ReLU]/d kappa3 =
-alpha*normal_pdf(alpha)/(6*v); the common1/6 does not change ordering.
This is a local, modeled, fixed-variance Gaussian proxy. It does not include
learned-head, covariance, star, radial, recurrent, later-layer or endpoint
derivatives. It is not an endpoint-risk Gram, tensor-error minimizer, or a
guarantee of accuracy improvement. Columns selected by this proxy receive
no further scaling. There is no response-mask restriction or random seed.

Use precisely the existing FP32 weighted-Gram application and single QR,
rank384/age4, signed physical coefficients, entry-frozen coefficient-square
weights, old-core updates, Lazy maps and schedule, active/birth/response0.01,
all learned literals, kernels and caches. Means/variances are already
available before each join; selection uses no labels. Pass indices through
one local state key and consume it once in the join. No selection or join
is added at terminal layer15. Preserve all other parent source bytes and
prove the inverse of both edits before any numerical run.

First run actual FNP helper certificates on fixed1024-unit synthetic
fixtures: zero and equal-score ties, mixed means/variances, symmetric means,
and tiny/clamped variances with underflowing tails. Require exact stable
index agreement with independent NumPy, 384 distinct in-range indices,
exact repeats, unchanged inputs, exact sign symmetry and first384 selection
for all-zero means. Verify the actual helper's two-axis FP32 gather against
NumPy for noncontiguous previous rows and exact parent gather in the zero
case. Finite-difference the third derivative of the analytic Gaussian ReLU
mean for nonzero standardized means; scaled error<=2e-5. Reject shape
mismatches. These tests do not claim preservation of the compressed tensor.

If certificates pass, run full native35/100/35 and independent cold100.
Require finite nonnegative16x1024 output, first5 parent rows byte-exact,
exact own repeated35 and warm/cold100 output files, retired numeric caches,
<8GB RSS and zero added swap. Seal complete predictions and bills before
opening the existing ordinary1B development means. Compare with exact Lazy
own-cold35/100. Advance only if both raw and adjusted scores improve on both
networks and pooled adjusted ratio<=.99, with adjusted=raw*max(.1,F/2^41).
These reused development networks are not independent generalization data.

On failure close this fixed selection without changing score, subset size,
join range, rank, QR, precision, learned head or fit/blend. On success seal
the wider panel/confirmation/client/contract/package gates before running
them. Future qualified Phase2 uploads are preauthorized by the user.

One nice10 BLAS1 numerical group at a time, existing8GB/256MiB-growth
watchdog; qualification requires zero added swap. Unit60s, native180s,
cold10090s, score30s. New artifacts<=15MB. No GPU, new sampling, training,
large downloads, or messages. Preserve all previous failed experiments.
