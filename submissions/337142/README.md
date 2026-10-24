# Submission #337142: young dead-row crop z 3.0 extended to layer 13 on #337126, graded 1.4328308e-9 (-0.027%)

2026-10-19T14:52:40Z · submission · adopted · measured

A superset of #337126's physics (per-layer _YNT_Z 3.0 at the layer-13 and layer-14 pushes, _OTT_Z 3.0 at layer 14) fired as a sub-step (local -0.095% vs #337126 on 32 ranks). Hosted VALID 100/100 at 1.432830842e-9, a new best by -0.027%: the layer-13 rung's local raw -0.16% transferred at about half (MSE -0.087% paired, 23/50 lower) after reading flat at 35 networks, which set the rule that raw-driven sub-steps no longer fire.

Recorded measurement:

    exact 1.432830842e-9 (-0.027% vs #337126; to J2W -0.902%, to oliver -1.010%, to 1.41e-9 -1.593%); paired 50 public nets vs #337126: MSE -0.087% (23/50 lower), flops +0.054%, adjusted -0.037%; util .109163, MSE 1.312e-8; public residual max .3685, private .360; predict wall max 112.0 s; early read at 35 nets MSE -0.045% (se .04, 14/35 lower), flops +0.049%, adjusted -0.003%; platform queue delay 32 min before the first evaluation record

Public score: 1.4328308421238806e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5964](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5964)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5971](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5971)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5978](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5978)
- [SUBMISSION_LEDGER.md#L1110](../../records/SUBMISSION_LEDGER.md#L1110)
- [SUBMISSION_LEDGER.md#L1111](../../records/SUBMISSION_LEDGER.md#L1111)

<details>
<summary>All result files and records (5)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [eval_337142.json](eval_337142.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Layer-13 dead-row crop at threshold 3.0: -0.095% adjusted; extending to layer 12 gives +0.013%](../../experiments/late-layer-exactness/layer-13-dead-row-crop-threshold-3-0-0/README.md)
- supersedes: [Submission #337126: layer-14 old-tail and young-crop exactness z 3.0 on #337090, graded 1.4332116e-9 (-0.100%)](../337126/README.md)
- ← supersedes: [Submission #337212: combined zero-cost dial optima on #337142, graded 1.4296259e-9](../337212/README.md)

[Research graph](../../graph/index.html#node=submission-337142) · [Copy provenance](../../PROVENANCE.json)

