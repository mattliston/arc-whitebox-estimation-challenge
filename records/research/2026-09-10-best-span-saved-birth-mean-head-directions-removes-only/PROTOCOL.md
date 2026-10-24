# Saved-direction magnitude diagnostic — 2026-09-09 PDT

This is one prospective, offline information test, not a candidate, fit,
deployment mixture, validation score, or reopening of an old estimator family.
Its purpose is to test whether simple rescaling of the already obtained P1 and
NODE+P1 output corrections could plausibly span the missing one-third final
error energy. It does not bound nonlinear retraining or new geometry suppliers.

Use exactly the 40 previously scored native VALID networks, excluding the
repeated rank37. Compare frozen d50, P1, and NODE+P1 outputs. Authenticate their
native acquisition results, completion seals, output hashes, and the two score
receipts; require the common P1 error and complete bill to match exactly between
the receipts. No truth arrays, model loading, estimator call, training, or
candidate parameter changes are permitted.

Let d1=P1-d50 and d2=NODE+P1-P1. For each network compute their final-layer
Gram matrix directly from the stored float32 outputs using float64 arithmetic.
Recover error inner products from the immutable same-truth MSEs by the identity
2<e,d>=MSE(e+d)-MSE(e)-<d,d>. For d2 relative to d50, subtract <d1,d2>.

Report exactly three unrestricted optimistic projections: d1 alone relative
to d50, d2 alone relative to P1, and their two-dimensional affine span relative
to d50. Aggregate both unweighted raw error and weighted error using the fixed
NODE+P1 per-network multipliers. These weights are only reference weights:
computing and combining multiple estimators is NOT billed here, so the
reference-weighted minimum is NOT a measured adjusted candidate score.

The projection coefficients use already-scored VALID error statistics. They
are diagnostic oracle values, are not deployable fitted coefficients, and will
not be emitted as a candidate. No per-network fitted coefficients, adaptive
subsets, additional directions, or follow-up amplitude sweep.

Prospective decision: if even the two-direction optimistic reference-weighted
minimum exceeds 3e-9, simple calibration of these corrections cannot establish
sub3 on this panel. If it is below, that establishes only a potential direction
span; an independently trained provider, complete actual compute, independent
14-net panels, production wall/residual checks, and official grading would
still be required. Small validated original components remain preserved.

Implementation is stdlib-only, nice10, one CPU process, 60-second hard alarm,
512-MiB RSS ceiling checked on every network, small fixed input set, exclusive
new output directory, finite/shape/domain checks and before/after hashes.
No MPS or external process. Old receipts and all frozen work remain unchanged.
