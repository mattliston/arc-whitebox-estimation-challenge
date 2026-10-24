# Submission #336754: collector disabled during prediction: residual spread halved, one network failed

2026-10-17T15:58:51Z · submission · failed · measured

The cyclic collector was disabled during prediction, one charged collection ran at the end, and survivors were frozen after each prediction. Residual median rose +3.5% versus #336741 while sd/median fell to 2.1% from 4.6%; one network failed at .418 s on a worker +6% slow.

Recorded measurement:

    Official public score 1.472541387307923e-09; failed networks 1; utilization 0.11211812275229022; wall mean/max 97.429/109.698 s; residual mean/max 0.372/0.418 s.

Public score: 1.472541387307923e-09. Status: GRADED. Failed networks: 1.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4879](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4879)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4885](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4885)
- [SUBMISSION_LEDGER.md#L1067](../../records/SUBMISSION_LEDGER.md#L1067)
- [SUBMISSION_LEDGER.md#L1068](../../records/SUBMISSION_LEDGER.md#L1068)
- [SUBMISSION_LEDGER.md#L1070](../../records/SUBMISSION_LEDGER.md#L1070)

<details>
<summary>All result files and records (8)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [eval_336754.json](eval_336754.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_summary.txt](eval_summary.txt) · record

</details>

[Research graph](../../graph/index.html#node=submission-336754) · [Copy provenance](../../PROVENANCE.json)

