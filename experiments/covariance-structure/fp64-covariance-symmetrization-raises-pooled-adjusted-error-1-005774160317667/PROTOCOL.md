# Fixed covariance symmetry restoration on Lazy

Phase 2 only, September 19 Pacific / September 20 UTC. The exact parent is
submitted Lazy b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c,
official #331599 adjusted 3.8164492657795035e-9. The approximately 1e-9 goal
remains active. This is a small numerical component, with no forecast of a
large accuracy gain and no claim of new distributional information.

The parent's FP32 fast covariance products are widened to FP64 and consumed
without an explicit symmetry projection. In real arithmetic a covariance and
both W.T C W and W.T (A C A) W are symmetric. Restore that identity by replacing
each full Cpre and Bpre by (X+X.T)/2 in FP64, immediately after their affine
acquisition and before every full-state consumer. At layer zero Bpre aliases
the same symmetrized Cpre. The final scalar-only diagonal route is unchanged.
The whole served covariance, tensor birth, trace and features consume the
same resulting matrices. No PSD repair, eigenvalue clipping, learned-mean
recenter change, tensor-plane symmetrization or added fitted parameter occurs.
K21 is a directed repeated-index plane and must NOT be symmetrized.

This only removes antisymmetric rounding. It does not certify that the
remaining symmetric component is accurate or PSD, nor that altered rounding
improves endpoint error. Earlier centered-feedback fitting, source additions,
pair-metric compression, and low-rank covariance experiments remain closed.
The inspected archive had symmetry in a different GMM4 reference; that failed
model is not evidence for this single current-bank numerical change.

Build with one insertion in the LAST _predict_binary and one inert helper;
prove an exact whole-source inverse. All literals, rank384/age4, Lazy schedule,
factor consumers, kernels, Weight5 lifecycle and hooks remain unchanged.
Freeze protocol, helper, build, candidate, unit, scorer and native runner
before full calls or opening reference arrays.

Independent small tests cover symmetric PSD, indefinite, deliberately skewed,
diagonal and zero FP64 inputs; check exact symmetry, diagonal preservation,
idempotence, input immutability, explicit entrywise reference, and quadratic
form preservation within 1e-12 scaled absolute error. The helper rejects
nonsquare, nonmatrix and non-FP64 arrays. Actual FNP arrays and bills repeat
exactly. No NumPy implementation is substituted into the estimator.

Then run native ranks35,100,35 and independent cold100 using the existing
weight5_reuse_v1/native.py. Require finite nonnegative 16x1024 outputs, exact
own repeat/warm-cold100 bytes, first-layer mean equality to Lazy, retired
cache, each process group below8GB and zero added system swap. Later selectors
may change; all scores use actual complete own cold FLOP bills. No numerical
identity with the old full output is claimed or required for an accuracy test.

Seal predictions and bills before reading the two pinned ordinary1B truth
files. Advance only if raw AND adjusted error improve on both networks and
pooled mean(raw*max(.1,own_F/2^41)) is <=.99 of exact Lazy. Failure closes this
fixed variant, with no strength/location/precision/kernel sweep or fitted
rescue. A pass requires a separately fixed wider panel and real NumPy-free
client/runtime, contract and archive checks before any submission. Future
qualified Phase2 uploads have the user's standing preauthorization.

One nice10 BLAS1 numerical process group. Existing8GB/256MiB-added-swap guard;
qualification requires zero added swap. Unit30s, three-call180s, cold10090s,
scoring30s. New artifacts<=10MB. No GPU, sampling or large dataset acquisition.
