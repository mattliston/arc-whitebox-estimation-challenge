# Submission #336389: census-refitted prebinding for the adaptive 128-column join

2026-10-16T03:52:20Z · submission · rejected · measured

The candidate retained #336381 predictions while reducing requests with a 20-network census-refitted prebind list. One hidden residual 0.426s failure did not invalidate its public score, which was +0.127% worse than #336342.

Recorded measurement:

    {"score_public": 1.4820703394703927e-09, "n": 100, "failed_networks": 1, "utilization": 0.11124306832305592, "wall_mean_s": 92.944, "wall_max_s": 104.409, "resid_mean_s": 0.36, "resid_max_s": 0.426}

Public score: 1.4820703394703927e-09. Status: GRADED. Failed networks: 1.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3815](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3815)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3884](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3884)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4119](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4119)
- [SUBMISSION_LEDGER.md#L1015](../../records/SUBMISSION_LEDGER.md#L1015)
- [SUBMISSION_LEDGER.md#L1023](../../records/SUBMISSION_LEDGER.md#L1023)
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
- [eval_336389.json](eval_336389.json) · result

</details>

## Related work

- ← shipped in: [Twenty-network prebind census refit: 836 fewer first and 295 fewer warm requests](../../experiments/setup-budget/twenty-network-prebind-census-refit-836-fewer-first-295/README.md)

[Research graph](../../graph/index.html#node=submission-336389) · [Copy provenance](../../PROVENANCE.json)

