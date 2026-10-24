# Score-aligned rank32 covariance representation audit

September6,2026, Phase2, mini57/post13 only. This is a representation
diagnostic, not another endpoint feature/loss fit or a retry of a closed
one-high third/fourth tensor carrier. The newly certified Gaussian score
metric materially changes the meaning of useful covariance directions.
Ask whether a small full-covariance correction is intrinsically plausible
under that metric before attempting any predictor.

Target E is TOTAL true post13 covariance minus the actual bank's post13
covariance after mean feedback and cold mask, not merely the isolated K21
pair response. Use the already downloaded100M public mini57 source's
post mean and M11 (already admitted for affine checks); no new download,
network, official1e9 mean score, final target, or sample. Population raw
moments are centered in f64 and symmetrized. Require raw-file hash and
metadata match, raw diagonal consistency, source post13->pre14 affine
covariance check <=1e-4 relative, and conservative f32-storage bound RMS
<=.005 of error RMS. The latter does not bound MC uncertainty; no replicate
or noise correction is claimed.

Reconstruct label-free bank geometry exactly as in the completed risk
audit and require the same six pre14 byte checks. A=I-beta uu^T and
Z=A^(1/2) E A^(1/2). Fix k=32 before labels. Compare exactly two bases:

1. Input-only: largest32 algebraic eigenvectors of
   A^(1/2) C_bank A^(1/2), no target in this choice.
2. Oracle comparator:32 eigenvectors of Z with largest absolute eigenvalue.
   This minimizes Frobenius truncation, not claimed globally optimal for
   the trace-augmented risk among all possible subspaces.

Within each fixed orthonormal basis Q, the exact trace-augmented metric
projection has core Q^T Z Q + c I, where
c=(tr(Z)-tr(Q^T Z Q))/(k+2). Lift through A^(-1/2) to get the rank32
physical covariance correction. Certify the fixed-subspace solution using
explicit small dense symmetric least squares before the wide test.

Require BOTH analytic risk and actual W14 next-mean response residual
ratios <=.25 for the input-only basis to consider a compact geometry-based
correction promising. Oracle results distinguish basis failure from a
more general rank bottleneck, but an oracle pass alone does not authorize
any predictor. Report unweighted covariance ratios and target/storage
diagnostics separately. No rank/frame/threshold sweep, fitted coefficient
predictor, state write, closed-loop raw/adjusted score or submission.
This is an in-sample noisy-truth oracle representation audit, not a final
accuracy ceiling or a proof about every rank32 approximation.
