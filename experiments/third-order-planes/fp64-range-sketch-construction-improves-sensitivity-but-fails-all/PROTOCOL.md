# FP64 shared-sketch native response intervention

Parent is exact official331684 (facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0).
The completed shared_sketch_roundoff_diagnostic_v1 measured4..5e-6
relative native sketch rounding and late projector differences up to21.24%
on rank38. At first changing layer10 the same-input FP64 projector response
was much smoother and smaller. This justifies a distinct FULL trajectory
intervention; it does not establish accuracy or revive any failed fit.

Change ONLY sketch formation at the existing ten joins5..14: cast the current
J, already-squared weight, Omega, Qp and core toFP64, compute the same
associated matrix formula using ordinary FNP matmul, and keep Y inFP64
through reduced QR. Round Q once toFP32; discard R. All core/factor updates,
carriers, old/young ranks, history ages, selection order, downstream kernels,
learned values and other precision remain the exact parent's. No clipping,
normalization, symmetrization, rank change or new source is added. The source
edit must invert to original parent and all six original head variants.

Units evaluate the actual new helper on all40 saved diagnostic sketch inputs
against an independent NumPy FP64 reference, plus tiny first/old/strided
cases. Require Q finiteFP32, orthogonality<=1e-5 and projector relative
error<=2e-6. Save actual helper bills; FP64 is paid, not a free reference.

Exactly fifteen cold full calls: bases36/38, plus/minus/double at head
coordinates0/8 on both, and36/head0plus repeat. Original epsilon1/16 is
unchanged. Base first5 rows must equal original parent; every probe first8
rows must equal its own NEW base, repeat all16 output bytes/full bill exact.
All arrays finite/nonnegative, all caches retired. Complete own-cold bills
must be<=1.20 of each corresponding original source call. A base cost
failure stops before acquiring responses. Sources, original controls and
the earlier diagnostic are sealed before execution; targets stay unopened.

For EACH of the same four cells, centered response must exceed20 times
combined FP32 output spacing; forward/center discrepancy<=.10 and double
curvature<=.10. These gates are unchanged from the old head-response study.
On failure close this intervention without epsilon/coordinate/precision/
scope rescue, training or upload. A positive result permits designing an
actual-cost supervised proposal and accuracy test, not an automatic bank.
No target scoring is part of this diagnostic. Its increased compute must
ultimately be repaid by measured raw-error reduction before submission.

One nice10 process tree, six BLAS limits1,8GB RSS,zero added swap. Units120s;
each full native call120s; response analysis90s. Reuse original datasets;
no downloads, new samples, GPU, dense tapes or fitting. Artifact cap50MB.
Preserve every prior failed experiment and official bank/version.
