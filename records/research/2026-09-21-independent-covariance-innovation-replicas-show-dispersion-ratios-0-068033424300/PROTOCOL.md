# Current-trajectory covariance-innovation replica audit

Phase2 only. Checkpoint82 completed concrete progress: the unchanged nonlinear
edge payload failed its fixed panel (adjusted1.009582713003627,5/14 wins).
Official331684 remains3.5844144212297072e-9; the full goal remains active.

Measure a missing property of the exact checkpoint70/81 supervision target,
using saved exact331684 rank35 states and the two existing independent8M
reference streams. The old replica audit used oracle previous states and is
not a noise measurement of this current-trajectory target. Its6.09% covariance
and9.23% next-response figures remain valid only for that old target.

Use exactly the prior Gaussian-gain local innovation at layers8..14:
U_s=sym(Ctrue_s-Cparent)-T_parent(mtrue_s_previous-mparent_previous,
sym(Ctrue_s_previous-Cparent_previous)). Copy tangent_cov byte-for-byte.
Both streams use the same frozen parent fields and canonical weights.
Next response is L(U)=pdf_next/(2sigma_next)*diag(W_next.T U W_next).
Reproduce original streamA wide label energies and stored FP32 U exactly.
Verify affine provenance of both reference streams at layer8<=1e-4 and the
linearity identity U_A-U_B=deltaC_current-T(deltaMean_previous,deltaC_previous).

Report each layer and pooled covariance, diagonal, off-diagonal and response:
observed=(mean(A*A)+mean(B*B))/2; dispersion=mean((A-B)^2)/2;
cross=mean(A*B); observed=dispersion+cross. No clipping of negative cross
or dispersion fractions above1. This is a two-stream point diagnostic,
not a confidence interval or a population error bound. Interpretation as
per-stream noise requires assumptions: both streams share MPS/FP32 bias,
and the fixed parent and fitted corrections have used streamA in training.
Conditioning on those fitted objects does not create an independent test.
Rank35 is a TRAIN network. Do not call either fitted-arm cross-risk unbiased,
held-out generalization, or a noise-corrected official score.

Before decoding either reference here, produce and seal the unchanged final
checkpoint81 output-only16 and nonlinear528 model deltas on the saved current
rank35 features. No new fit, normalizer, model selection or coefficient.
Use the exact copied FP32 forward and independent explicit-channel FP64
reference; source<=3e-5,delta<=1e-4; bank reconstruction byte-exact. Measure
each fixed correction against A and B and report cross residuals. The exact
gain difference must equal2*mean(delta*(U_A-U_B)), including the response
channel. This describes sensitivity to reference reuse, not generalization.

Synthetic units cover the noise decomposition, independent orthogonal noise,
zero and anticorrelated streams, correction/gain identities, diagonal versus
off-diagonal accounting, and the tangent against an explicit scalar-indexed
reference. Float64 identity tolerances2e-10 relative with exact-zero handling.
After measurement, independently reaggregate all saved full labels/deltas
using dot products and fsum; require agreement<=2e-10. Also check16 fixed
next-response columns with separate vector-matrix-vector contractions.

Prospective decision: if pooled dispersion/observed>=.20 in covariance OR
next response, further independent-label acquisition merits a separately
designed study before another fit. Otherwise this audit does not support
sample noise as the main explanation for the failed current fits. No new
reference generation, rank100 expansion, seed/sample-size sweep, fitting,
candidate, panel, runtime, package or upload follows automatically either
outcome. No existing negative or milestone is changed.

One serial nice10 BLAS1 process group,8GB RSS ceiling,256MiB emergency swap
stop,ZERO added swap required. Units90s,predictions180s,measurement180s,
independent audit120s. Expected RAM<2GB,new artifacts<=256MB. No native
estimator calls,new samples,downloads or GPU work. Preserve sources and all
failures; never restart a live controller on an observation timeout.
