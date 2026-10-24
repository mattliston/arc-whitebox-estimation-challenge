# Fixed paired-probe correction: REJECTED, closed

Measured September 7, 2026, 06:21:49–06:21:51 PDT (actual receipt mtimes).
Phase 2 ONLY. No candidate, truth comparison, training or submission.

The complete prescribed eight groups ran once: development observer_fit
ranks35 and100, four independent groups each, N32 per group, FP64,
two independent one-Gaussian-probe replicas sharing the same actual rows.
Reference/control/cyclic law and seeds are frozen in the protocol/runner.
This closes that configuration; no extra groups, tuning or larger rollout.

## New measurement and decision

Mean paired-probe statistic Y =1.753658848299398.
Conservative necessary-condition ceiling T =3.278400781631656e-5.
Prospectively fixed rejection threshold20T =0.0006556801563263312.
Decision: REJECT_FIXED_DEV2_NECESSARY_CONDITION.

Y/T =53,491.289354396875 is an observed statistic ratio, NOT a measured
candidate adjusted-score ratio. The sole preregistered rejection test has
ideal-law false-rejection probability at most.05 under E(meanY)<=T.
The corresponding observed-threshold Markov tail bound is1.8694632566709704e-5,
subject to the stated floating-point/PRNG qualifications.

This statistic includes cancellation across all16 correction sites.
It is a probe-only contribution to cyclic sampling variance, not target
mean error, total variance, a score, or an independent14-net result.
It diagnoses only the fixed FP64 reference/control/probe/ordinary-dense
circuit on these two development networks, not all Mehler constructions
or an untested FP32 implementation.

The cost threshold gives every operation except the two ordinary dense
forwards free. Those forwards alone cost67,076,096 FP32-convention FLOPs
per row. Increasing the sample count of this same fixed law does not
remove its variance-times-work requirement. No complete candidate bill
was measured, and no competitive raw-accuracy result follows.

## Numerical/resource receipt

All eight groups and16-layer checks completed. All arrays/statistics were
finite; maximum actual standardized deviation5.392405463885469, within
the declared domain limits. All source/input hashes and mtimes unchanged.
Prior297-point100-decimal Stein and exact tiny JVP/quadratic/affine checks
passed; these are finite numerical checks, not uniform analytic bounds.

One nice10 CPU worker, explicit one-thread environment, no torch/MPS or
threadpoolctl import. Wall2.83890129160136s, peakRSS479,100,928 bytes,
both well below180s/2GiB. Diagnostic literal FP64 GEMM arithmetic subtotal
149,177,237,504 FLOPs; other work excluded from that subtotal but included
in measured wall/RSS. No full-candidate cost claim.

Swap before/after784.75MiB used of2048MiB. Our worker exited0. A separate
transient Python3.13 process was observed and left untouched; do not claim
the entire machine was idle or that it was our numerical worker.

## Frozen receipts

- Protocol: `2c557fbd3607193b8636dd052de056f4c9395aa6151a96c76068fd8148167375`.
- Core: `507293c3efab1c299d6c43628709ef9aad51ddfbae64c101e9efcbe77a186b49`.
- Tiny qualifier: `d2040f4a786f0d49310511ce3480a0cc11a71ef7e84fd8b77f48efeea7b57bb8`.
- Tiny JSON: `5e69f79244f5afb8ea9abba357b365ad189737c02d5c1afbd370382f39f88c97`.
- Wide runner: `ca3e62eb5a67ea56f2ff74bbcbb09b9e9f82f695cc6f7914eb671df3b9572c53`.
- `paired_mehler_probe_rejection_v1/started.json`:
  `362572e0639975e8c8c45864ba6cb87f10b99c9a8b1f9d14c69c499e03dddb2f`.
- `paired_mehler_probe_rejection_v1/result.json`:
  `b97f0d3b9f8633b758a181588f15d56655995c145b47cc51e426e9d9a1e4464f`.

Read `PROVENANCE_AND_COST_CORRECTION_2026_09_07_V1.md` before reusing older
notes: historical observer_fit35/MLP89 IS today's development rank35,
although its unconditional post-ReLU gradient summary is the wrong
statistic for this correction. That additive correction also supersedes
the earlier nominal leading-arithmetic cost ceiling. Old receipts remain.

## Score/release state remains unchanged

Confirmed official bank #329926: adjusted4.982231460183304e-9.
Retained response-row component: independent14-net adjusted4.83424098592912e-9,
paired improvement3.25073% versus its bank comparator4.996669003386079e-9.
Candidate and panel-result hashes rechecked unchanged after this screen.
No <4e-9 milestone, submission or promotion. The response exception is
complete and retained, not reopened. Final goal strictly<1e-9, with
validated milestone banks<4e-9/<3e-9/<2e-9, remains active.
