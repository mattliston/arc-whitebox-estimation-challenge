# Submission #337126: layer-14 old-tail and young-crop exactness z 3.0 on #337090, graded 1.4332116e-9 (-0.100%)

2026-10-19T13:48:43Z · submission · adopted · measured

Fired as a sub-step and as a calibration of raw-driven transfer: per-layer _OTT_Z 3.0 and _YNT_Z 3.0 at the layer-14 push on the #337090 file (local raw -0.35%, bill +0.20%, adjusted about -0.15% on 32 ranks, no request cost). Hosted VALID 100/100 at 1.433211582e-09, -0.100% vs #337090: MSE -0.268% paired on the 50 public networks (3/4 of the local raw) and util +0.18% (0.9 of the local bill).

Recorded measurement:

    exact 1.433211582e-09 (-0.100% vs #337090; to J2W -0.928%, to oliver -1.036%, to 1.41e-9 -1.620%); util .109104 (+0.18%); MSE 1.313e-8 (-0.268% paired, 31/50 lower); public residual median .3549 / max .3804, private max .364; predict wall max 113.3 s; early read 25 nets: MSE -0.261% (se .11, 14/25 lower), flops +0.179%, adjusted -0.077% (se .10); gates setup5 3.58 s, stage A 23 signatures; RSS 7.5-7.6 GB

Public score: 1.4332115821132753e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5955](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5955)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5956](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5956)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5960](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5960)
- [SUBMISSION_LEDGER.md#L1108](../../records/SUBMISSION_LEDGER.md#L1108)
- [SUBMISSION_LEDGER.md#L1109](../../records/SUBMISSION_LEDGER.md#L1109)

<details>
<summary>All result files and records (5)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [eval_337126.json](eval_337126.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Layer-14 exactness (old tail z 3.0 + young crop z 3.0): raw -0.352%, bill +0.270%, adjusted -0.079% on 28 ranks](../../experiments/late-layer-exactness/layer-14-exactness-old-tail-z-3-0-young/README.md)
- supersedes: [Submission #337090: final-layer third-cumulant row threshold 3.0 and own prebinding list](../337090/README.md)
- ← supersedes: [Submission #337142: young dead-row crop z 3.0 extended to layer 13 on #337126, graded 1.4328308e-9 (-0.027%)](../337142/README.md)

[Research graph](../../graph/index.html#node=submission-337126) · [Copy provenance](../../PROVENANCE.json)

