# Submission #336657: parent executors for push products and three reads: two warm residual failures

2026-10-17T05:24:38Z · submission · failed · measured

All 18 registered push-peeling routes and three read routes reverted to parent executors, with cold preparation moved into setup. All five cold networks passed, but two warm public networks failed at .405196953 and .424088030 s residual.

Recorded measurement:

    Official public score 0.04980302713774312; failed networks 2; utilization 0.11209036914658099; wall mean/max 98.572/110.143 s; residual mean/max 0.341/0.424 s.

Public score: 0.04980302713774312. Status: GRADED. Failed networks: 2.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4826](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4826)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4828](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4828)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4830](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4830)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4832](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4832)
- [SUBMISSION_LEDGER.md#L1048](../../records/SUBMISSION_LEDGER.md#L1048)
- [SUBMISSION_LEDGER.md#L1050](../../records/SUBMISSION_LEDGER.md#L1050)
- [SUBMISSION_LEDGER.md#L1052](../../records/SUBMISSION_LEDGER.md#L1052)

<details>
<summary>All result files and records (7)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_336657_from_saved_snapshot_v1.json](eval_336657_from_saved_snapshot_v1.json) · result

</details>

## Related work

- ← shipped in: [All push-peeling routes and three reads reverted to parent executors: cold failures removed](../../experiments/arithmetic-kernels/all-push-peeling-routes-three-reads-reverted-parent-executors/README.md)

[Research graph](../../graph/index.html#node=submission-336657) · [Copy provenance](../../PROVENANCE.json)

