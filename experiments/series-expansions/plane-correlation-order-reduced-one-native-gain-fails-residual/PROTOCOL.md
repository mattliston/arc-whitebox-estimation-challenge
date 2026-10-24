# Prospective isolated plane-bond E1 component (not execution admission)

Parent is frozen Gaussian3/precision/batched-symmetry candidate88683ee1,
955150 bytes. Six count-checked source edits replace only the pair-plane
correlation sum r=0,1,2 by r=0,1 and the matching active GN covariance read.
Reverse the records in build.json to recover every original parent byte.
No released source or existing qualification changes.

For derivative-channel arrays F/G, the five plane contractions are
sum_{r=0..E} rho^r/r! F[a+r] G[b+r], with (a,b)=(2,1),(1,2),
(2,2),(3,1),(1,3). E changes2 to1 in all three pair outputs. Gaussian R3,
local EL2, derivatives0..6, scalar/marginal arithmetic, learned literals,
normalization/centralization, symmetry routes and all setup keys remain fixed.
L0's separate Gaussian path and L15's terminal scalar source remain unchanged;
require exact L0 output against the parent. Later outputs may change.

The GN correction is a separately implemented linear read of those same plane
inputs. Its t21a/t31a must lose their rho^2/2 terms too: otherwise the effective
trained covariance coefficients would differ at the dropped order. The active
GN path receives the just-published pair aux/Cpre identity at layers2..14.
Its original fallback is preserved; this candidate is scoped to that normal
full-model path. No coefficient is refitted or changed. Retain EL2: the old
local-order1 test had about +0.3% raw loss.

Historical evidence is modest, not a forecast. fable_analyst_v1/audit_s4.py:57–73
defines the6:1:2 experiment; audit_s4_v1.jsonl:65–68 records final MSE
1.570517693107198e-8,1.580770762955408e-8,1.758562003739315e-8,
1.4694059778123538e-8 on ranks35,100,37,39, versus the file's four base rows.
These correspond approximately to -.10%,+.13%,+.008%,+.055% raw.
RESULT.md:745–805 flagged its small cost benefit; RESULT.md:1528 says lower
series flags were not rerun on S23. gn3_knobs.py:28–44 implements GN by a second
call to that modified pair map, motivating the current explicit GN match.
This is not a strong accuracy-positive prior. Higher-order removal can disturb
learned compensation and change discrete downstream selections.

Source-priced dominant saving at width1024: X2 shrinks45 to30 planes. Two
multiplies plus its sum save45*n^2 per call,660602880F over14 calls. Removing
rho2 and two GN outer/multiply/add tails saves8*n^2 per call,109051904F over13
calls. Total769654784F (about0.28% of the historical276.55GF bill), before small
gather tariffs and later-route effects. One X2 view bind saves15728640F.
The shared arena allocation and T/U offsets do not shrink; no RSS gain is
claimed. No additional matrix products or input-dependent scopes are added.

Root may separately admit the source-bound tiny probe under60s/1GB/zeroSwap,
BLAS1. It extracts the actual X2 and GN statements (no estimator import), tests
three dyadic signed/zero-correlation cases and all three channels against
independent FP64 E1 sums, and verifies E2-minus-E1 equals the omitted term.
It tests GN input-perturbation equivalence and exact parent inversion. No probe,
native job, broad run, or release is authorized by this draft.

Future native acquisition uses the existing native_probe.py and frozen native
screen engine1fc332a5, with exact parent88683 as component comparator. Fixed
TRAIN2 ranks35,100: strict adjusted improvement versus parent,2/2 wins, all16
finite and L0 byte-exact. If admitted, early two-network boundary feasibility
precedes fixed TRAIN6 ranks36,45,52,58,64,70. Combined TRAIN8 requires >=.05%
component adjusted gain,>=6/8 wins, paired50k bootstrap seed20260926 CI upper<0.
No raw veto, added-scope/layer/order/coefficient variant, or failure rescue.
Resources remain150s/300s,7.5GB/zeroSwap/BLAS1; setup682/649+9<=4.1s.

Bank admission remains a separate latest-official matched VALID40 comparison:
>=1% adjusted gain,>=28/40 wins, CI upper<0,max per-network regression<=10%,
then actual serving qualification. The release candidate and official baseline
must be pinned prospectively then; this component receipt cannot grant release.
