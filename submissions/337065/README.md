# Submission #337065: join sketch columns 1536/1024/1024 on #337061, graded 1.4390249e-9 (-0.252%, full bill transfer)

2026-10-19T07:10:58Z · submission · adopted · measured

The release form of the column squeeze (#337061 base + _JSK_KC {7: 1536, 10: 1024, 13: 1024}) graded VALID 100/100 at 1.439024868e-9, the second bank step (< 1.440e-9): util -0.30% vs #337061 (the local -0.297% transferred in full) while the local raw -0.09% did not (MSE +0.04%). The public residual maximum reached .3972 of the .40 cap and the predict wall 117.6 s of 120.

Recorded measurement:

    exact 1.439024868e-9 = -0.252% vs #337061 (1.4427189e-9), -0.313% vs #337028; util .109351 (-0.30%); MSE 1.316e-8 (+0.04%); public residual median .3492 / p90 .3662 / max .3972; predict wall public max 117.6 s, private max 106.2 s; early read on 10 public nets -0.22% (util -0.27%, MSE +0.05%); gates setup5 3.32 s, stage A 23 signatures

Public score: 1.4390248682398952e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5922](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5922)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5923](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5923)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5925](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5925)
- [SUBMISSION_LEDGER.md#L1104](../../records/SUBMISSION_LEDGER.md#L1104)
- [SUBMISSION_LEDGER.md#L1105](../../records/SUBMISSION_LEDGER.md#L1105)

<details>
<summary>All result files and records (5)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [eval_337065.json](eval_337065.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Join-sketch columns reduced to 1536/1024/1024: -0.390% adjusted; smaller counts lose raw accuracy](../../experiments/join-sketch/join-sketch-columns-reduced-1536-1024-1024-0-390/README.md)
- supersedes: [Submission #337061: level fusion, prebinding refit and two operand box views, graded 1.4427189e-9](../337061/README.md)
- ← supersedes: [Submission #337090: final-layer third-cumulant row threshold 3.0 and own prebinding list](../337090/README.md)

[Research graph](../../graph/index.html#node=submission-337065) · [Copy provenance](../../PROVENANCE.json)

