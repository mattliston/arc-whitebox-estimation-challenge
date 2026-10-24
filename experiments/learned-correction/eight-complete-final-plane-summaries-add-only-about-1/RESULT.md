# Eight complete final-plane summaries: insufficient additional information

The fixed information screen fails on both existing development networks,
even with separate privileged coefficients for each network and zero added
inference cost. The added summaries make a small positive contribution, but
do not satisfy either advancement threshold. No producer, cross-network fit,
candidate or upload is selected from this study.

| Existing 1B network | Ideal / parent | Ideal / affine control | Served / parent | Served / affine control |
| --- | ---: | ---: | ---: | ---: |
| 35 | 0.880978601109 | 0.992102057743 | 0.878351394848 | 0.991217255851 |
| 100 | 0.883441280077 | 0.987926830649 | 0.880118318074 | 0.986550088563 |

Each network required both ideal and served ratios <=0.85 against the
unchanged parent and <=0.90 against the matching affine oracle. All eight
comparisons fail. Ideal extra improvement beyond affine is 0.790% / 1.207%;
served extra improvement is 0.878% / 1.345%. The roughly 12% reduction against
the parent includes the privileged refit of its existing 65-column readout.
It cannot be credited to the eight added summaries or called a deployable gain.
These reused references are not held-out validation or official scores.

The eight summaries are row mean, RMS and mean absolute covariance
correlation; row/column norms and sums of the standardized repeated-third
plane; and correlation times alpha. Every retained signed factor, full star,
trace term, row and diagonal is included. The frozen feature definitions are
in PROTOCOL.md. Postactivation covariance summaries are excluded. This study
asks a different question from the previous 65 hidden-direction queries,
while reusing their authenticated exact-331684 captures. No new physical
cumulant source or universal representation limit is established.

Both controls have full rank (65 and 73). Independent QR solves using
separately accumulated column scales and losses reproduce the SVD solutions,
all clamped FP32 prediction bytes and every gate decision. Maximum coefficient
relative difference is 2.0828350495e-14;
maximum ideal prediction relative difference is
1.38051725916e-17. Privileged coefficients
remain explicitly labeled ORACLE_DO_NOT_DEPLOY.npz. Ideal predictions are
unclipped FP64; served predictions are nonnegative FP32.

Six small explicit-tensor/indexed-congruence fixtures verify all eight
statistics, including nonsymmetric inputs, no factors, signed cancellation,
zero terms and permutation. Nonzero references use true relative errors;
exact-zero references use absolute errors. Maximum error is
6.22282396967e-16 against 2e-10. Wide independent 8-by-8 plane
and covariance slices agree within 1.93450976983e-15
and 4.13865854891e-16, respectively.
Formal FP64 diagonals differ from native k3 by at most
7.02414581569e-07; covariance diagonals differ
from the original sigma squared by at most 2.63095526419e-06.
Those checks meet their fixed 1e-3 tolerances; they do not assert exact equality
to the native packed arithmetic. Inputs are unchanged and 35/100/35 feature
bytes repeat exactly. Features were sealed before reference decoding.

The optimistic screen retains each parent's independent cold multiplier:
413448696909 F for 35 and 416264792412 F for 100. The old warm capture's
100 bill is not used as its cold cost. Research FP64 feature contraction
has no measured deployment bill or runtime qualification; zero extra cost is
an explicit screening assumption, not a native implementation claim.

All four guarded groups retired; controller 56034 is terminal. Zero new
native estimator calls, new samples, downloads or GPU calls. Two existing
capture files (229523060 B) were reused, not copied.
There were four same-network least-squares projections and four independent
QR replays, with zero cross-network fits. Peak sampled group RSS was
1040105472 B, self-reported peak
1082097664 B, with zero added swap. Fast-stage
sampling can understate memory; use self-report or the whole-study maximum.
Artifacts before closeout total about 1.542 MB.

This closes these eight summaries and the declared linear information
question, without a feature, normalization, rank, SVD or nonlinear rescue.
All earlier negative scopes remain intact. Official submission 331684 is
unchanged at adjusted 3.5844144212297072e-9, 100/100 complete, zero failures.
The 4e-9 crossing is banked; 3e-9, 2e-9 and approximately 1e-9 remain unmet.
The goal remains active.
