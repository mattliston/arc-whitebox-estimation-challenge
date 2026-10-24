# Submission #336678: freeze the setup object graph: one public residual failure

2026-10-17T07:28:05Z · submission · failed · measured

Setup-owned closure cycles were broken and the live setup graph was collected and frozen while ordinary automatic collection remained enabled. Prediction bytes and FLOPs matched on ten natural-lifecycle cases, but public network 43 failed at .403555464 s residual.

Recorded measurement:

    Official public score 0.027370444789183063; failed networks 1; utilization 0.11208239560335642; wall mean/max 97.482/107.991 s; residual mean/max 0.337/0.404 s.

Public score: 0.027370444789183063. Status: GRADED. Failed networks: 1.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4837](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4837)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4839](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4839)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4842](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4842)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4850](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4850)
- [SUBMISSION_LEDGER.md#L1052](../../records/SUBMISSION_LEDGER.md#L1052)
- [SUBMISSION_LEDGER.md#L1054](../../records/SUBMISSION_LEDGER.md#L1054)
- [SUBMISSION_LEDGER.md#L1056](../../records/SUBMISSION_LEDGER.md#L1056)
- [SUBMISSION_LEDGER.md#L1058](../../records/SUBMISSION_LEDGER.md#L1058)

<details>
<summary>All result files and records (6)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result

</details>

## Related work

- ← shipped in: [Setup closure-cycle removal and heap freezing cut maximum full-collection pause by 58.90%](../../experiments/wall-and-residual/setup-closure-cycle-removal-heap-freezing-cut-maximum-full/README.md)

[Research graph](../../graph/index.html#node=submission-336678) · [Copy provenance](../../PROVENANCE.json)

