# Submission #336649: coalesced temporary-view cleanup: two cold public networks exceeded runtime caps

2026-10-17T04:44:16Z · submission · failed · measured

Coalescing temporary-view cleanup removed 2961 hidden free requests and passed five concurrent starts at 4.857–4.876 s with exact prediction and FLOP parity. Platform setup passed, but one cold public network exceeded 120 s wall time and another exceeded the 0.4 s residual cap.

Recorded measurement:

    Official public score 0.030238825302653264; failed networks 2; utilization 0.1117641890282357; wall mean/max 102.011/113.539 s; residual mean/max 0.341/0.412 s.

Public score: 0.030238825302653264. Status: GRADED. Failed networks: 2.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4813](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4813)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4815](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4815)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4819](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4819)
- [SUBMISSION_LEDGER.md#L1044](../../records/SUBMISSION_LEDGER.md#L1044)
- [SUBMISSION_LEDGER.md#L1046](../../records/SUBMISSION_LEDGER.md#L1046)
- [SUBMISSION_LEDGER.md#L1048](../../records/SUBMISSION_LEDGER.md#L1048)

<details>
<summary>All result files and records (9)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [eval_336649.json](eval_336649.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_summary.txt](eval_summary.txt) · record
- [eval_336649_partial_v1.json](eval_336649_partial_v1.json) · result

</details>

## Related work

- ← shipped in: [Concurrent official startup reproduced a setup failure hidden by isolated qualification](../../experiments/setup-budget/concurrent-official-startup-reproduced-setup-failure-hidden-isolated/README.md)

[Research graph](../../graph/index.html#node=submission-336649) · [Copy provenance](../../PROVENANCE.json)

