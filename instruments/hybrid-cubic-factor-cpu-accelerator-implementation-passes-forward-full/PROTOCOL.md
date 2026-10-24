# CPU moments with MPS carrier products: a new numerical qualification

Prospective, 2026-09-09 UTC / September8 Pacific. Phase2 only. No fit,
new targets, native candidate, score or submission is authorized by this file.
All older sources and failures remain frozen.

## Two-route decision and score objective

Confirmed bank330390 is4.53886349950395e-9 adjusted,2.120838555441651e-8
raw. Its exact d50 source has local independent14 adjusted4.4837991190464774e-9,
raw2.103192289821446e-8 and mean complete467682934475.3571 FLOPs.

1. Selected eligibility question: can autonomous current-P adaptation use a
   numerically faithful CPU64-moment/MPS32-carrier implementation? The current
   bank's60 birth coefficients improved paired native40 raw about5.19%
   (adjusted about4.89%); the whole-package official comparison improved8.899%.
   Its
   current P remains teacher-mean-reset trained. Additional P gain is UNKNOWN;
   this is evidence of an untested training configuration, not a12% forecast.
   Same-shaped P replacement adds no inference operators at fixed geometry.
   About11.9% raw improvement would cross4e-9,33.9% would cross3e-9 at unchanged
   bills. Changed masks require the eventual candidate's OWN complete bill.
   Approximate inference cost remains467.68B, not a new measurement. The
   unresolved assumption is learnable autonomous error; this qualification
   tests only numerical eligibility and resource/throughput, NOT that value.
2. Difference-leg-only CP compression has a concrete old-frame constructor,
   but its optimistic historical net saving was4.0--4.36%, after40--41.6B
   acquisition/restoration, with the old physical plane credited free. Current
   d50 does not have that full physical plane free. There is no current
   endpoint raw result, and old difference-map spectra were not more
   compressible than common bases. Even unchanged raw and4% lower complete
   cost would only give roughly4.36e-9. Cheapest useful future test is a paid
   current constructor/restoration census plus consumer-error witness; that
   audit does not currently justify a substantial compression run.

The raw route is retained because even the.1 multiplier floor requires raw
below1e-8 to reach the final strict1e-9 goal. All eventual advancement uses
mean_i(raw_i * max(.1, complete_F_i/2^41)); no proxy-times-unrelated-bill claim.

## Exact implementation difference

The previous all32-MPS current-P attempt FAILED its fixed5% scalar secant
gate before gradients or training. It remains failed; no threshold, metric,
seed, amplitude, or checkpoint from it is changed or selected here.

Construct a new adapter using its unchanged CPU PCard and the pinned BD step.
Keep every moment, scalar map, nonlinear pair calculation, head, selector,
birth action, star, covariance push, parameter tensor and reduction on CPU
with the CPU parent's original dtypes. Replace ONLY these two FP32 products:

    pp = w.T @ p
    subplane = ((left*left) @ (right*lam).T).to(dt)

with differentiable CPU->MPS matmul->CPU calls. CPU squaring and weighting
remain outside the helper. No detach, custom VJP, approximation, extra
clipping, operator reassociation, selector freeze/order change or fallback.
Empty products remain CPU, preserving their exact zero-width behavior.
Source inverse must exactly recover the unchanged observed CPU step.

This explicitly different precision placement removes all32-MPS moment/head
arithmetic, but MPS GEMM and its backward may still change the result. There
is no presumed numerical pass. It changes OFFLINE execution, not the bank.

## One fixed forward stage

Authenticate the old passed CPU-forward receipt6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607
and every reused artifact before decoding. Reuse its actual FP32 null and
+/-.025,+/-.05 payloads and original cotangents on TRAIN35/100. Do not
generate a different direction or scalar. CPU served-output reduction stays
float64 exactly as in that receipt, including the final FP32 output cast.

For each rank, first disable both offloads and require all16 null bytes AND
all ordered selector records byte-equal to the saved CPU PCard. Then execute
all five fixed hybrid cases. Require finite nonnegative16x1024 FP32 outputs,
all16 max absolute difference<=1e-5 versus the SAME saved CPU case, finite
prevariances>=1e-10 with zero floor events, and all47 selector observations.
Record between-implementation membership AND ordering differences, but do
not freeze them. Each rank's two symmetric scalar secants must agree within
5% using denominator max(abs(CPU),abs(hybrid),1e-10). Save before fail-stop.

PASS admits only the separately executed CPU/hybrid complete-P VJP stage
described below. FAIL closes this fixed hybrid implementation; no additional
offload sites, precision/schedule/seed/amplitude sweep or fit follows.

## Gradient stage, conditional on forward PASS

For both same ranks, compare the complete45200 allocated /45134 active P
coordinates at null. Same scalar/cotangent, exact cstar, original dynamic
selectors and non-reentrant layer checkpointing. CPU and hybrid checkpointed
primals must each be byte-identical to their own saved null forward. Require
full gradient relative L2<=5%; each PKEY with CPU norm>=1e-6 of full norm
must pass relative5%, smaller groups absolute discrepancy<=5e-8 of full norm.
All gradients finite, layer0 head/bias/gain gradients exactly zero, frozen
normalizers without gradients. Every observed replay must retain exact
prevariance bytes and ordered selectors; do not manufacture replay visits.
No target arrays or optimizer exist in this qualification.

Report forward and forward/backward wall, CPU RSS, MPS driver sampling and
explicit forward-helper transfer volume, including checkpoint replays but
excluding automatic backward copies and allocator staging. These decide
whether a later fixed fit is affordable; no
training budget or fit permission is inferred from a numerical pass. Native
d50 perturbed-P primal/secant compatibility also remains a distinct missing
bridge before any fit/release claim. Old CO downstream drafts are not resumed.

## Resources and preservation

One nice10 process, BLAS1, sole MPS job, no fallback.900s hard independent
wall limit per stage,12GB host/tree RSS,16GB sampled MPS driver, stop on
>2GiB swap growth. Source hashes fixed before launching; exclusive output
directories, failure outputs preserved. Independent watchdog plus cooperative
driver checks; sampled memory is not an absolute-peak guarantee.
No numerical execution until main reads implementation and independent review.
Keep official release limits8GB client+server,120s perMLP,.4s residual and
conservative local<60s unchanged for any FUTURE candidate. No forum posting,
commits, promotion, root-estimator change or frozen-file edit.
