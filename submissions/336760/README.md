# Submission #336760: two-strip covariance sandwich: 100 networks passed with .327 s median residual

2026-10-17T16:40:33Z · submission · adopted · measured

The two-strip covariance sandwich was added to per-prediction heap freezing and the dtype trim. It removed 3,109 requests per network for +0.66% local bill; all 100 networks passed with residual median .327 s and maximum .344 s.

Recorded measurement:

    Official public score 1.4812993823293046e-09; failed networks 0; utilization 0.1128687709886799; wall mean/max 95.603/107.973 s; residual mean/max 0.327/0.344 s.

Public score: 1.4812993823293046e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4885](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4885)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4886](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4886)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4895](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4895)
- [SUBMISSION_LEDGER.md#L1072](../../records/SUBMISSION_LEDGER.md#L1072)
- [SUBMISSION_LEDGER.md#L1073](../../records/SUBMISSION_LEDGER.md#L1073)
- [SUBMISSION_LEDGER.md#L1075](../../records/SUBMISSION_LEDGER.md#L1075)
- [SUBMISSION_LEDGER.md#L1077](../../records/SUBMISSION_LEDGER.md#L1077)

<details>
<summary>All result files and records (8)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [eval_336760.json](eval_336760.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_summary.txt](eval_summary.txt) · record

</details>

[Research graph](../../graph/index.html#node=submission-336760) · [Copy provenance](../../PROVENANCE.json)

