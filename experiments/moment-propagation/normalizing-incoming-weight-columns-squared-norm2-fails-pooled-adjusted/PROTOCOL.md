# Fixed incoming-column normalization of the same ReLU network

Prospective Phase2 protocol,2026-09-19 PDT. Submitted Lazy source
b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c
is the sole parent. No numerical or accuracy result for this candidate has
been opened. Prior Krylov, pair-metric, covariance symmetry, tensor sketch,
sampling and trained-cstar failures remain preserved and closed.

Question: does a fixed canonical hidden-unit scaling improve this estimator's
actual objective? Positive diagonal hidden scalings leave a bias-free ReLU
network unchanged in exact arithmetic. The approximate radial projection,
shared compression metric, learned weight features and finite-precision
execution are not covariant under that change. This is a new approximation
test, not a bug fix, added physical information, or a forecast of1e-9.
The earlier GAUGE_RADIAL_PROJECTION_INTERPRETATION_V1.md already explains
why natural-coordinate radial projection has a valid isotropic-query meaning.
This test does not assert that canonical coordinates are necessarily better.

Fix D_-1=I. For layers0..14, compute V_l=D_(l-1)^-1 W_l and
D_l,jj=sqrt(2/sum_i V_l,ij^2). If a column is exactly zero, use D_l,jj=1.
Use W'_l=V_l D_l. At layer15 set D_15=I and W'_15=D_14^-1 W_15.
All scale/reweight arithmetic is FNP FP64, with each transformed weight
stored FP32 before the original estimator. Carry the FP64 D values, not
reconstructed or retuned scales. Restore every reported row by division
by D_l and round to FP32; final-row D is exactly one. No clipping, scale
strength, damping, norm target selection, sweeps, fitting or averaging.

In exact arithmetic h'_l=D_l h_l by positive homogeneity, including final
output equality. Stored FP32 weight rounding means this is not an assertion
of bit-identical network arithmetic. Verify the actual implementation first.
The transformation itself uses weights only, no target/current estimator
state. It preserves width/depth and runs the entire unchanged original
estimator on the transformed weights. Every transformation, cast, copy,
output restoration and changed selector/shape cost must be charged.

Source: change only the final Estimator.predict call to an inert new wrapper;
append normalization helpers/proxy class; prove a whole-source byte inverse.
Retain the original shape/budget fallback and all learned literals, physics,
rank/age/Lazy schedules and kernels. Existing numerical-cache ownership and
retirement must work on the proxy. No original input mutation or persistent
transformed-weight cache is permitted.

Fixed actual-FNP synthetic unit cases, seed2026092006: He-scaled32-wide5-layer
weights, those with nonuniform positive column scales, and explicit zero
columns. Compare direct dense original/transformed ReLU trajectories on17
fixed inputs, restoring each layer. Require scaled maximum error<=3e-6;
weight reconstruction error<=2e-7, nonzero hidden squared-column norms within
2e-6 of2, finite positive FP64 scales, finite FP32 weights, exact final scales,
input bytes unchanged, exact repeated arrays and full unit bills. Wrong shape
and nonfinite inputs must reject. These fixtures validate the coordinate
identity, not the approximate estimator's invariance or accuracy.

If units pass: native35/100/35, then independent cold100 under existing guards.
Require finite nonnegative16x1024 output, exact own repeat and warm/cold100
parity, original cache retirement, <8GB owned RSS, zero added swap. Restored
first-layer means must be within2e-5 maximum absolute difference of exact
saved Lazy; do NOT require deeper means or first-layer byte equality.
Save and seal candidate predictions and complete bills before opening the
ordinary1B35/100 truth means. Compare both candidate cold starts with exact
saved Lazy cold outputs. Advance only if BOTH raw and adjusted scores improve
on BOTH networks AND pooled mean(raw*max(.1,ownF/2^41))<=.99 of Lazy.
These are reused development networks, not independent generalization.

Failure closes this fixed normalization without scale/target/layer/order/
precision/kernel/head/metric/fit rescue or blend. On success predeclare wider
paired validation and actual client/contract/package before any banking claim.
Future qualified Phase2 uploads have user standing authorization. This is no
authorization to submit an unqualified or failed candidate.

One nice10 BLAS1 numerical group,8GB process-tree guard,256MiB added-swap
hard stop, zero added swap required for qualification. Unit60s, native180s,
cold10090s, scorer30s. New artifacts<=15MB; no GPU or large download. Preserve
all existing files, root estimator and official Lazy archive.
