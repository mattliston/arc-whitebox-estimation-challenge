# Submission #336759: per-prediction heap freeze and dtype comparisons: 100 networks passed

2026-10-17T16:33:29Z · submission · adopted · measured

Per-prediction freezing retained young-generation collections and bounded the charged end scan, while dtype comparisons removed string conversion. Predictions, FLOPs and requests were identical to #336741; all 100 networks passed with residual median .349 s and maximum .366 s.

Recorded measurement:

    Official public score 1.472645587066553e-09; failed networks 0; utilization 0.11212886088471351; wall mean/max 96.951/108.13 s; residual mean/max 0.349/0.366 s.

Public score: 1.472645587066553e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4883](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4883)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4884](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4884)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4890](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4890)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4893](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4893)
- [SUBMISSION_LEDGER.md#L1072](../../records/SUBMISSION_LEDGER.md#L1072)
- [SUBMISSION_LEDGER.md#L1073](../../records/SUBMISSION_LEDGER.md#L1073)
- [SUBMISSION_LEDGER.md#L1075](../../records/SUBMISSION_LEDGER.md#L1075)

<details>
<summary>All result files and records (8)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [eval_336759.json](eval_336759.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_summary.txt](eval_summary.txt) · record

</details>

## Related work

- ← shipped in: [Charged end scans grew with worker age; per-prediction freezing kept scans at 0.8–2.9 ms](../../records/research/2026-10-17-charged-end-scans-grew-worker-age-per-prediction-freezing/README.md)

[Research graph](../../graph/index.html#node=submission-336759) · [Copy provenance](../../PROVENANCE.json)

