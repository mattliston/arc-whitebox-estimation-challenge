# Submission #336381: one adaptive 128-column join block on compact kernels

2026-10-16T03:21:33Z · submission · rejected · measured

A static prefix plus one greedy 128-column join block was submitted on the lean compact-layout parent. Hidden network 58 exceeded residual 0.412s; all 50 public networks scored normally but aggregate +0.159% versus #336342 was worse.

Recorded measurement:

    {"score_public": 1.4825492381044546e-09, "n": 100, "failed_networks": 1, "utilization": 0.11128425716421589, "wall_mean_s": 93.001, "wall_max_s": 102.787, "resid_mean_s": 0.36, "resid_max_s": 0.412}

Public score: 1.4825492381044546e-09. Status: GRADED. Failed networks: 1.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3773](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3773)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3852](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3852)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4119](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4119)
- [SUBMISSION_LEDGER.md#L1013](../../records/SUBMISSION_LEDGER.md#L1013)
- [SUBMISSION_LEDGER.md#L1019](../../records/SUBMISSION_LEDGER.md#L1019)
- [SUBMISSION_LEDGER.md#L1028](../../records/SUBMISSION_LEDGER.md#L1028)

<details>
<summary>All result files and records (10)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ROOT_FIRE_RECEIPT_v1.json](ROOT_FIRE_RECEIPT_v1.json) · result
- [ROOT_PUBLIC_GRADE_v1.json](ROOT_PUBLIC_GRADE_v1.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [validate.json](validate.json) · record
- [stage_a.json](stage_a.json) · record
- [package.json](package.json) · record
- [validate_package.json](validate_package.json) · record
- [eval_336381.json](eval_336381.json) · result

</details>

## Related work

- ← shipped in: [Removing low-value grouped-read plans: worst wall 109.0 to 103.6 seconds for +0.09% bill](../../experiments/wall-and-residual/removing-low-value-grouped-read-plans-worst-wall-109/README.md)
- ← shipped in: [Static-plus-one adaptive 128-column join block: pilot adjusted -0.475%, request risk 19.64%](../../experiments/join-sketch/static-plus-one-adaptive-128-column-join-block-pilot/README.md)

[Research graph](../../graph/index.html#node=submission-336381) · [Copy provenance](../../PROVENANCE.json)

