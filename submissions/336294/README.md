# Submission #336294: Hermite covariance grouping and QR read with exact view caching

2026-10-15T18:37:01Z · submission · adopted · measured

Hermite covariance group fusion combined with the signed-prefix power-step and QR read, then view-only request repair, improved the official public score by -0.532713% versus #336178. All 100 networks completed without failure.

Recorded measurement:

    {"score_public": 1.4890381684157092e-09, "n": 100, "failed_networks": 0, "utilization": 0.11122980429864583, "wall_mean_s": 85.537, "wall_max_s": 92.217, "resid_mean_s": 0.353, "resid_max_s": 0.375}

Public score: 1.4890381684157092e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3472](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3472)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3487](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3487)
- [SUBMISSION_LEDGER.md#L993](../../records/SUBMISSION_LEDGER.md#L993)
- [SUBMISSION_LEDGER.md#L995](../../records/SUBMISSION_LEDGER.md#L995)

<details>
<summary>All result files and records (14)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ROOT_PREFIRE_RECEIPT_v1.json](ROOT_PREFIRE_RECEIPT_v1.json) · result
- [ROOT_PREFIRE_VIEW_RECEIPT_v1.json](ROOT_PREFIRE_VIEW_RECEIPT_v1.json) · result
- [ROOT_PUBLIC_GRADE_v1.json](ROOT_PUBLIC_GRADE_v1.json) · result
- [ROOT_FIRE_v1.json](ROOT_FIRE_v1.json) · result
- [ROOT_PREFIRE_v1.txt](ROOT_PREFIRE_v1.txt) · result
- [ROOT_PREFIRE_VIEW_v1.txt](ROOT_PREFIRE_VIEW_v1.txt) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [validate.json](validate.json) · record
- [stage_a.json](stage_a.json) · record
- [package.json](package.json) · record
- [validate_package.json](validate_package.json) · record
- [eval_336294.json](eval_336294.json) · result

</details>

## Related work

- ← shipped in: [Hermite covariance grouping into two products per order: all-24 adjusted -0.570899%](../../experiments/covariance-structure/hermite-covariance-grouping-into-two-products-per-order-all/README.md)
- ← shipped in: [Hermite grouping exceeds the request-risk gate at 26.17%; view caching repairs it to 19.35%](../../records/research/2026-10-15-hermite-grouping-exceeds-request-risk-gate-26-17-view/README.md)
- ← shipped in: [Signed-prefix power step and QR on the completed young read: all-24 adjusted -0.069%, held for stacking](../../experiments/third-order-planes/signed-prefix-power-step-qr-completed-young-read-all/README.md)
- ← enables: [Submission #336178: young-read rank-two tail completion with request repair on #336125, bill -1.34%, 1.49701e-9](../336178/README.md)

[Research graph](../../graph/index.html#node=submission-336294) · [Copy provenance](../../PROVENANCE.json)

