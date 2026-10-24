# Direct native covariance-head optimization

Declared before this study decodes any target or evaluates any perturbed loss.
Parent is official331684, adjusted3.5844144212297072e-9 with100/100 complete,
source facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0.
The separate native linear-response arm failed all4 sampled linearity/curvature
gates and remains closed. Its predictions, secants, and unopened targets are
not used to fit here. Exact unchanged-parent replay excluded stale baselines.

This distinct method measures the actual nonlinear final loss at every update.
Only the16 shared covariance output coefficients may change. All trunk weights,
normalizers, bias, source formula, layer window8..14, full-lower physics,
readout, timers and accounting remain byte exact. No extra production work.

Use one fixed projected SPSA run:36 iterations, six independently permuted
cycles of TRAIN[36,38,40,41,43,44], PCG64 seed2026092066. Each iteration uses a
single same-network plus/minus pair with independent Bernoulli signs in16
coordinates. For zero-based t, a_t=.1*(4.6/(t+4.6))**.602 and
c_t=(1/16)/(t+1)**.101. Loss is own-cold adjusted final-layer MSE divided by
the unchanged parent's adjusted loss on the same network, using existing
ordinary1B means. Gradient=(Lplus-Lminus)/(2*c_t)*delta. Clip its Euclidean
norm at1, update the FP64 center, project into the radius.75 Euclidean ball
around the original FP32 head. Probes are unprojected center+/-c_t*delta,
rounded once to FP32; their displacement is bounded by1.000001. Preserve all
sources, outputs, bills, losses, gradients and centers. No line search, restarts,
extra iterations, best-checkpoint selection, alternate seed or gain rescue.
Deployment is the FP32-rounded mean of the final18 post-update centers.

SPSA formula source: James Spall, JHU APL,
https://www.jhuapl.edu/spsa/PDF-SPSA/Matlab-SPSA_Alg.pdf
The finite schedule, clipping, projection, averaging and constants above are
task choices. No convergence theorem is claimed for this discontinuous native
map. Tiny tests must recover an independent quadratic's exact gradient after
averaging all16 Hadamard directions, and verify projection, gains, FP32 source
rounding and unchanged non-head constants. The prior actual FNP covariance
kernel certificate remains applicable because the kernel is unchanged.

Before fitting, reuse the freshly qualified exact-parent cold36/38 audit and
run four fresh packed-parent calls40/41/43/44. Require all16 output bytes and
complete cold bills identical to the pinned existing uncompressed-parent calls.
Seal schedule, six baselines and all sources before opening TRAIN targets.
Every new candidate call must be finite, nonnegative16x1024 FP32, first8 rows
byte exact to parent, on the identical weight hash, with complete own-cold bill
and retired caches. Failures stop without algorithm rescue.

Evaluate the one terminal averaged head on all6 TRAIN networks with fresh
cold calls. Require pooled adjusted ratio<=.995, at least4 wins and worst<=1.05.
If positive, CHECK35/100 is disjoint from this refit, but reused research and
original covariance pretraining, not pristine. Run fresh packed-parent cold
35/100, candidate35/100/35 plus independent cold100. Require first8 parent
bytes, exact candidate repeats and warm/cold predictions, own-cold pooled
adjusted<1 and each<=1.05. Compare scores using complete independent cold bills.

Only if CHECK passes, fixed panel[45,46,47,49,50,51,52,53,54,55] needs pooled<1,
at least6wins and worst<=1.10; fixed confirmation[37,39,42,48,59,63,66,67]
needs pooled<1,at least5wins,worst<=1.10. Existing frozen covariance native
outputs/bills are legitimate parent comparisons: packing changed only exact
constant representation. No tuning follows any gate. These are reused public
research networks; any positive result still requires an official grade.

Release requires exact ordinary NumPy-free client/native parity and complete
bills, cold and prediction-only timing scopes recorded separately, every local
candidate residual<=.25s, matched total-wall pooled<=1.075 and each<=1.10,
contract/package checks and exact source/archive pins. Qualified uploads use
standing authorization. Official completion and adjusted score decide banking.

One nice10 numerical worker at a time, each supervised at8GB RSS, six BLAS
limits1 and zero added swap. A lightweight supervised controller may wait for
its single child group; do not double-count its elapsed time. Each cold call
120s, warm triple240s, controller3600s. At most120 full calls for this arm,
including release; artifact cap150MB. No new samples, datasets, GPU or dense
state tapes. All previous failures and immutable source/result records remain.
