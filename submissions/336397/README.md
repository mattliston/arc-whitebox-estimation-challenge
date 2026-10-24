# Submission #336397: three greedy 128-column join blocks with prebind refit; three residual failures

2026-10-16T04:34:15Z · submission · failed · measured

Block-greedy pivots with own prebinding were submitted after 24-network confirmation. Networks 1/49/71 exceeded residual limits;48 completed public networks showed MSE +0.01%, bill +0.66% and aggregate +0.61%, so the local gain did not transfer.

Recorded measurement:

    {"score_public": 0.028816917171601, "n": 100, "failed_networks": 3, "utilization": 0.11147543009840774, "wall_mean_s": 92.697, "wall_max_s": 105.858, "resid_mean_s": 0.361, "resid_max_s": 0.42}

Public score: 0.028816917171601. Status: GRADED. Failed networks: 3.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3866](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3866)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3938](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3938)
- [SUBMISSION_LEDGER.md#L1021](../../records/SUBMISSION_LEDGER.md#L1021)
- [SUBMISSION_LEDGER.md#L1026](../../records/SUBMISSION_LEDGER.md#L1026)

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
- [eval_336397.json](eval_336397.json) · result

</details>

## Related work

- ← shipped in: [Block-greedy join pivots: pilot adjusted -0.760%, but request risk 30.95%](../../experiments/join-sketch/block-greedy-join-pivots-pilot-adjusted-0-760-but/README.md)
- ← shipped in: [Full census refit makes the block-greedy join release pass: request risk 23.10 to 17.21%](../../experiments/setup-budget/full-census-refit-makes-block-greedy-join-release-pass/README.md)

[Research graph](../../graph/index.html#node=submission-336397) · [Copy provenance](../../PROVENANCE.json)

