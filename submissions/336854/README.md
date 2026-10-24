# Submission #336854: heavy-column Nyström join sketch: 100 networks passed at 1.4517480806360376e-09

2026-10-18T02:15:20Z · submission · adopted · measured

The Nyström basis sketch used the heaviest 2048/1536/1536 joined columns at layers 7/10/13 while preserving full-column factors and rescaling the core trace. All 100 networks passed with residual median .345 s, p95 .365 s and maximum .391 s.

Recorded measurement:

    Official public score 1.4517480806360376e-09; failed networks 0; utilization 0.1103403133181564; wall mean/max 97.48/114.343 s; residual mean/max 0.347/0.391 s.

Public score: 1.4517480806360376e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5065](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5065)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5067](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5067)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5129](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5129)
- [SUBMISSION_LEDGER.md#L1081](../../records/SUBMISSION_LEDGER.md#L1081)
- [SUBMISSION_LEDGER.md#L1082](../../records/SUBMISSION_LEDGER.md#L1082)
- [SUBMISSION_LEDGER.md#L1083](../../records/SUBMISSION_LEDGER.md#L1083)
- [SUBMISSION_LEDGER.md#L1084](../../records/SUBMISSION_LEDGER.md#L1084)

<details>
<summary>All result files and records (8)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [eval_336854.json](eval_336854.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_summary.txt](eval_summary.txt) · record

</details>

## Related work

- ← shipped in: [Join sketch on 2048/1536/1536 heavy columns: 32-network adjusted and bill both -1.653%](../../experiments/join-sketch/join-sketch-2048-1536-1536-heavy-columns-32-network/README.md)

[Research graph](../../graph/index.html#node=submission-336854) · [Copy provenance](../../PROVENANCE.json)

