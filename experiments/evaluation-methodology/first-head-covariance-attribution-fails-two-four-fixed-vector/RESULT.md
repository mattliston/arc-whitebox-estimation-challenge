# First-head covariance attribution: completed, gate FAILED

2026-09-09 07:31 UTC / September 9 Pacific. Phase 2, target-blind diagnostic.
All four stages completed with numerical/integrity gates intact. The fixed
information gate failed. Do not promote this to a covariance implementation,
full-card qualification, gradient test, fit, raw/adjusted score, or submission.
The older BR and NG failures remain unchanged.

## Exact scope and results

Native N: all five sealed P cases on TRAIN35 and100, plus35 null repeat.
All two-row outputs reproduced the corresponding saved native bank prefixes
byte-for-byte. CPU: the same ten two-step cases on the identical paid native
G0; all35 saved NG prefix bytes and ordered selectors reproduced exactly.
Controls: native L and R, all five cases/both ranks, plus35 null L/R repeats.
Only Cpre/Bpre at layer1 changed; logical depth remained16 and downstream
layer1 physics/head were recomputed. No layer2+, gradients, P updates,
targets, MPS, fitting, candidate building or scoring occurred.

D is the native-N minus original-NG-CPU paired served layer1 response. E is
native-N minus native-L on the same native C0/W1. Gate: both norm(D-E)/norm(D)
and abs(scalar(D-E))/abs(scalar(D)) <=.5 on both ranks/both amplitudes, with
the predeclared nontrivial floors. Scalar is exactly the original helper's
layer1 cotangent contribution; it is not final-layer MSE.

| Rank | Amplitude | Remaining vector ratio | Remaining scalar ratio | Gate |
| --- | --- | --- | --- | --- |
| 35 | .025 | .4974263464 | .6208350235 | FAIL |
| 35 | .05 | .4228644706 | .0061162675 | PASS |
| 100 | .025 | .4924508346 | .1394479381 | PASS |
| 100 | .05 | .4927801134 | .9738723037 | FAIL |

All four vector checks passed; two scalar checks failed. That partial result
is preserved, not retuned or relabelled. R was diagnostic-only and did not
pass the vector gate in any case (remaining ratios .9422,.9759,1.0410,.9756).
No post-hoc circuit selection or extra amplitude was made.

## What it establishes, and does not

Native and CPU post0 centered covariance differ only at FP64 diagonal
roundoff: maximum3.3306690738754696e-16 on both ranks. Their FP32 cast bytes
are exactly identical. Applying the CPU left circuit to either C0 gives
exactly identical output. Thus post0 recentering/input-law mismatch is not
the first covariance push's source of discrepancy in these two cases.

For identical native C0/W1, N-L matrix Frobenius differences are
0.0009305160935 (35) and0.0009449198981 (100); maxima8.69855e-6/8.04243e-6.
R-L Frobenius differences are6.51903e-5/6.50964e-5. N-R includes backend,
layout and packed arithmetic differences; it is not pure Winograd causality.

Changing that covariance circuit influences served finite-difference vectors
materially, but does not meet the prospective scalar attribution criterion.
The actual pre-head host is P-independent in every arm (paired host response
zero). Served-minus-unrounded-response norms are roughly1.58e-5..1.67e-5
at .025 and7.73e-6..8.28e-6 at .05. These are measured rounding effects,
not permission to replace the original full-card numeric gate. Neither a
terminal-scalar repair nor any additional raw accuracy is established.

## Complete diagnostic cost/resource record

One nice10 CPU numerical child at a time, BLAS1, no MPS. Whole-sequence
watch58.8689327091s; maximum sampled combined watch+child-tree RSS
2,765,406,208B, under4,000,000,000B. Zero observed swap growth, no watchdog
termination, no remaining numerical child. Native N child12.6850s,
CPU13.7025s, native controls21.9971s, judge3.9281s; supervisor also included
source/admission/artifact verification and all post-context serialization.

Actual FNP diagnostic-context FLOPs: N cold13,399,308,660; each remaining
N12,157,962,492. Controls cold10,989,853,044; each remaining control
9,748,506,876. Bills include owned inputs, setup, repeated P initialization
and materialization. L/R arithmetic acquired in the CPU stage is additional
diagnostic work, not free deployment computation. These partial-prefix bills
must not be used as complete inference-cost savings or adjusted-score evidence.

## Frozen receipts

- N result `0cc32573881c8e63e49118460fa57d983653c069e14ff15db915db115769fd59`
- N completion `44e18e1ceec23451c0ffd02359dc3891589524076c6b46b5b0a1d9adba6ebebe`
- CPU result `660b35dd426d06ce46a1162cb50c002ee30db4bc629cbeff66b5109e0f7caa36`
- CPU completion `7fab8c1539672ce25d9ff1e560d0ba5ad85a43f71fef3c64aa3035bbda096531`
- Controls result `0ef450da95c54be5c19da33ca826c810d7d81e8a4098ccbd90c9c8f429e07009`
- Controls completion `9314e572c9568753ae7a3a6968be2d9d786a8b874bda185b5fc02344658ac6d4`
- Judge result `db86ec983c629d098a1f0093575fb9c83de924b51600201623bd52c8bcb1e6ed`
- Judge completion `b6071e5dda7e644b02ce817543f5754fa7b88fd221502a36dc0ed6d3f8806be9`
- Watch result `2827ac3041239cf6ad9efcfa00177ce339610283a458a2b9268cbdba3c438793`
- Watch completion `bcb890d7869946f0f6d0c27376b13d192f1429a37abe2c4be71ce19499581836`

Root separately verified all five completion seals and807 unique recorded
source/input/artifact hashes after completion using stdlib-only file hashing.
Judge and watch are completed=true/passed=false with no execution error.
Exit1 is the completed scientific rejection, not a grader failure.
All executable sources and run directories are now frozen; no retry.
