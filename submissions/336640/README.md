# Submission #336640: fixed 224-key setup and batched aliases: setup failure before evaluation

2026-10-17T03:38:47Z · submission · failed · measured

The setup repair fixed generic prebinding at 224 keys and batched 723 alias calls while retaining predictions and tariffs. Isolated startup passed at 4.162473 s, but five concurrent official starts later failed at 5.017–5.022 s; the platform exposed no traceback.

Recorded measurement:

    The submission failed during setup; no per-network score, utilization, wall or residual results were reported.

Public score: None. Status: FAILED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4785](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4785)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4789](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4789)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4809](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4809)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4811](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4811)
- [SUBMISSION_LEDGER.md#L1040](../../records/SUBMISSION_LEDGER.md#L1040)
- [SUBMISSION_LEDGER.md#L1042](../../records/SUBMISSION_LEDGER.md#L1042)
- [SUBMISSION_LEDGER.md#L1044](../../records/SUBMISSION_LEDGER.md#L1044)

<details>
<summary>All result files and records (8)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [eval_336640.json](eval_336640.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_summary.txt](eval_summary.txt) · record

</details>

## Related work

- ← shipped in: [Concurrent official startup reproduced a setup failure hidden by isolated qualification](../../experiments/setup-budget/concurrent-official-startup-reproduced-setup-failure-hidden-isolated/README.md)

[Research graph](../../graph/index.html#node=submission-336640) · [Copy provenance](../../PROVENANCE.json)

