# Submission #336925: 300-request product cap with complete startup checks: failed setup on three workers

2026-10-18T10:21:48Z · submission · failed · measured

The repaired portfolio used a 300-request cap and passed complete local five-worker startup qualification. On the platform two workers completed setup at 4.732193 and 4.742557 s, while three had no completed setup record; zero networks were evaluated.

Recorded measurement:

    The submission failed during setup; no per-network score, utilization, wall or residual results were reported.

Public score: None. Status: FAILED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5540](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5540)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5544](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5544)
- [SUBMISSION_LEDGER.md#L1092](../../records/SUBMISSION_LEDGER.md#L1092)
- [SUBMISSION_LEDGER.md#L1094](../../records/SUBMISSION_LEDGER.md#L1094)

<details>
<summary>All result files and records (9)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [eval_336925.json](eval_336925.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [public_status_v1.json](public_status_v1.json) · result
- [eval_summary.txt](eval_summary.txt) · record

</details>

## Related work

- ← shipped in: [Product request cap reduced from 500 to 300: -0.4553932186% adjusted on 32 networks](../../experiments/arithmetic-kernels/product-request-cap-reduced-500-300-0-4553932186-adjusted/README.md)

[Research graph](../../graph/index.html#node=submission-336925) · [Copy provenance](../../PROVENANCE.json)

