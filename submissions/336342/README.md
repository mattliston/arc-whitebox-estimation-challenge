# Submission #336342: grouped rank-40 read kernels with corrected whole-call billing

2026-10-15T23:38:18Z · submission · adopted · measured

Fixed 105-shape grouped strided-read kernel dispatch was submitted after correcting 276903104 omitted first-view FLOPs and captured-setup parity. The official score improved -0.407797% versus #336334 with 100/100 zero failures.

Recorded measurement:

    {"score_public": 1.480205182599705e-09, "n": 100, "failed_networks": 0, "utilization": 0.11073828900745866, "wall_mean_s": 96.672, "wall_max_s": 112.049, "resid_mean_s": 0.364, "resid_max_s": 0.397}

Public score: 1.480205182599705e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3575](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3575)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3589](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3589)
- [SUBMISSION_LEDGER.md#L1001](../../records/SUBMISSION_LEDGER.md#L1001)
- [SUBMISSION_LEDGER.md#L1003](../../records/SUBMISSION_LEDGER.md#L1003)

<details>
<summary>All result files and records (11)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ROOT_FIRE_RECEIPT_v1.json](ROOT_FIRE_RECEIPT_v1.json) · result
- [ROOT_PUBLIC_GRADE_v1.json](ROOT_PUBLIC_GRADE_v1.json) · result
- [ROOT_PREFIRE_v1.json](ROOT_PREFIRE_v1.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [validate.json](validate.json) · record
- [stage_a.json](stage_a.json) · record
- [package.json](package.json) · record
- [validate_package.json](validate_package.json) · record
- [eval_336342.json](eval_336342.json) · result

</details>

## Related work

- ← shipped in: [Grouped rank-40 matrix kernels: corrected all-24 adjusted -0.706639% versus the row-energy parent](../../experiments/arithmetic-kernels/grouped-rank-40-matrix-kernels-corrected-all-24-adjusted/README.md)

[Research graph](../../graph/index.html#node=submission-336342) · [Copy provenance](../../PROVENANCE.json)

