# Submission #336392: one adaptive 256-column join block with census-refitted prebinding

2026-10-16T04:16:04Z · submission · rejected · measured

The greedy 256-column block and source-specific prebind list passed release gates, but the apparent eight-network gain failed broader confirmation. Two hidden residual failures at 0.405/0.401s left a valid public score that was about +0.26% worse than #336342.

Recorded measurement:

    {"score_public": 1.4842280050093728e-09, "n": 100, "failed_networks": 2, "utilization": 0.11107239724849023, "wall_mean_s": 92.252, "wall_max_s": 106.592, "resid_mean_s": 0.358, "resid_max_s": 0.405}

Public score: 1.4842280050093728e-09. Status: GRADED. Failed networks: 2.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3845](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3845)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3910](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3910)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4119](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4119)
- [SUBMISSION_LEDGER.md#L1017](../../records/SUBMISSION_LEDGER.md#L1017)
- [SUBMISSION_LEDGER.md#L1025](../../records/SUBMISSION_LEDGER.md#L1025)
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
- [eval_336392.json](eval_336392.json) · result

</details>

## Related work

- ← shipped in: [Static plus one 256-column greedy block: eight-network adjusted -0.655%](../../experiments/join-sketch/static-plus-one-256-column-greedy-block-eight-network/README.md)

[Research graph](../../graph/index.html#node=submission-336392) · [Copy provenance](../../PROVENANCE.json)

