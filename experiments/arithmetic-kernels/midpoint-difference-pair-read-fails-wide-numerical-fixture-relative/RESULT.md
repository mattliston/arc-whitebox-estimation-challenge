# Fixed signed-pair read fails numerical qualification

The midpoint/difference read using the unchanged matrix kernel fails its
prospective wide-fixture tolerance. Relative error is **0.0035854277148522043**
against **0.00003**, about **119.5 times** the limit. This closes the fixed
candidate before any full-network call or score. The official bank remains
331684: adjusted 3.5844144212297072e-9, raw 1.949715441540434e-8,
100/100 cases complete, zero failures. The 4e-9 milestone is banked;
3e-9, 2e-9 and approximately 1e-9 remain unmet. The full goal stays active.

For each original signed pair, the candidate evaluates the exact real
identity a*(x*x*u-y*y*v) = (h*h+d*d/4)*(a*D)+(2*h*d)*(a*H),
where h=(x+y)/2, d=x-y, H=(u+v)/2 and D=u-v. It packs the same
contracted width and uses the original product function. All new arithmetic
is FP32. The common old basis, lazy map and addition order are retained.
Marginal cubic diagonals remain on their existing FP64 path. Factor
construction, compression, rank, cohort schedule, covariance, heads and
learned parameters are unchanged. The candidate is an exact parent prefix
plus an append-only override; its bytes did not change after qualification
started. No dominant product is removed by this identity.

The first unit attempt exposed an oracle defect: an exactly cancelling
fixture had exactly zero candidate and independent scalar outputs, while
the dense FP64 reference left a 5.679343126843894e-16 roundoff norm.
Dividing by that artifact produced an invalid relative failure. The
preserved diagnostic proves the exact pair cancellation. An additive
unit_v2 handles that declared exact-zero case, bounds the dense roundoff,
and requires exact zero from both candidate and scalar oracle. Reversing
the recorded edits recovers unit.py exactly. The candidate, fixture order,
random stream, precision, all nonzero tolerances and objective gates were
unchanged. Both failed source versions and logs remain preserved.

The repaired unit then failed on the fixed 512x2048 by 2048x512 close-pair
fixture. Five preceding small fixtures passed, including exact cancellation,
zero coefficients, multiple signed cohorts, repetition, permutation and
input preservation. The materialized/lazy/complete ownership checks and
malformed-input rejection checks occur after the wide fixture and were
never reached. There is no successful complete unit receipt.

A bounded diagnostic replayed the same fixture and compared the four
arithmetic paths below. Its FP64 polarization identity agrees with direct
FP64 evaluation to relative 5.478158045314982e-13.

| Synthetic fixture path | Relative error | Measured FLOPs |
| --- | ---: | ---: |
| Pair rewrite, existing kernel | 0.003585427714852204 | 730,838,014 |
| Original signed read, existing kernel | 0.01217643682958431 | 675,172,352 |
| Pair rewrite, direct matmul diagnostic | 4.206151103946926e-07 | 1,082,919,934 |
| Original signed read, direct matmul diagnostic | 0.0003895404827650046 | 1,075,576,832 |

The rewrite improves this synthetic error relative to the original path
but still fails its absolute gate. Direct matmul with the same pair helper
greatly reduces the error, identifying a substantial contribution from the
existing fast-kernel arithmetic on this fixture. These are synthetic kernel
measurements, not full-estimator accuracy or cost evidence. Direct products
are diagnostic comparisons only: no replacement candidate, new packing,
precision, layer choice, fit, or gate relaxation was introduced.

No planned disabled/enabled/cold full-network sequence ran; no truth or
weight arrays were decoded. Previously recorded input files were hash
verified only. No target score, panel, confirmation, package or upload was
produced. The prepared native and score scripts remain as unexecuted history.

Four guarded groups retired; controllers 5738, 89525, 17099 and 51028 are
terminal. Peak sampled group RSS was 71,352,320
bytes, and the wide diagnostic reported a process high-water mark of
271,073,280 bytes. Failed unit processes did
not write their own high-water marks; the sampled values are not exact peak
bounds. All four guards recorded zero added swap. Numerical work used
nice10, one BLAS thread and the existing 8GB group limit. About
0.707 MB was preserved before
closeout. No new samples, training, GPU work or downloads occurred.

The separate qualified covariance/readout archive remains unchanged.
Automatic approval review rejected its AIcrowd upload because it could not
verify trusted authorization for that exact private payload and destination.
The existing exact-upload approval question remains unanswered. No retry,
bypass, upload process or submission receipt occurred in this study.
