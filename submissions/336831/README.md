# Submission #336831: union push products for all roster shapes: three residual failures

2026-10-18T00:00:36Z · submission · failed · measured

All-shape union products reduced local request count, but a slow platform worker exhausted residual margin. Three networks failed, including public index 4 at .405 s and hidden index 63 at .414 s; median wall was 100 s and maximum 118.4 s.

Recorded measurement:

    Official public score 0.02039582873404175; failed networks 3; utilization 0.11195046715079116; wall mean/max 100.929/118.396 s; residual mean/max 0.352/0.414 s.

Public score: 0.02039582873404175. Status: GRADED. Failed networks: 3.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5011](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5011)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5030](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5030)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5044](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5044)
- [SUBMISSION_LEDGER.md#L1076](../../records/SUBMISSION_LEDGER.md#L1076)
- [SUBMISSION_LEDGER.md#L1078](../../records/SUBMISSION_LEDGER.md#L1078)
- [SUBMISSION_LEDGER.md#L1079](../../records/SUBMISSION_LEDGER.md#L1079)
- [SUBMISSION_LEDGER.md#L1081](../../records/SUBMISSION_LEDGER.md#L1081)

<details>
<summary>All result files and records (8)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [eval_336831.json](eval_336831.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_summary.txt](eval_summary.txt) · record

</details>

## Related work

- ← shipped in: [Removing the union-product shape guard reduced requests; the extra rank-48 kernel had no covered shapes](../../experiments/arithmetic-kernels/removing-union-product-shape-guard-reduced-requests-extra-rank/README.md)

[Research graph](../../graph/index.html#node=submission-336831) · [Copy provenance](../../PROVENANCE.json)

