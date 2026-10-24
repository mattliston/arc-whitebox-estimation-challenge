# Submission #336687: charged cleanup and adaptive late direct products: all networks rescued by a meter bug

2026-10-17T08:35:40Z · submission · adopted · measured

The runtime repair used charged full-generation cleanup and direct dense products at layers 12–14 when projected residual exceeded .370 s. The native SDK lacked the assumed active-context accessor, so unavailable context activated rescue on all 100 networks and added 7.77–9.35% bill.

Recorded measurement:

    Official public score 1.5984331261131061e-09; failed networks 0; utilization 0.12169566372813279; wall mean/max 91.944/103.363 s; residual mean/max 0.344/0.38 s.

Public score: 1.5984331261131061e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4856](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4856)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4858](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4858)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4861](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4861)
- [SUBMISSION_LEDGER.md#L1062](../../records/SUBMISSION_LEDGER.md#L1062)
- [SUBMISSION_LEDGER.md#L1064](../../records/SUBMISSION_LEDGER.md#L1064)
- [SUBMISSION_LEDGER.md#L1065](../../records/SUBMISSION_LEDGER.md#L1065)
- [SUBMISSION_LEDGER.md#L1067](../../records/SUBMISSION_LEDGER.md#L1067)

<details>
<summary>All result files and records (6)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result

</details>

[Research graph](../../graph/index.html#node=submission-336687) · [Copy provenance](../../PROVENANCE.json)

