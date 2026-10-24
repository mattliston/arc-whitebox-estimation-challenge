# Submission #336412: rank-384 stack with own prebind census, accumulator refit and least-recently-used caches

2026-10-16T05:06:54Z · submission · adopted · measured

The exact request repair preserved #336367 predictions and cut 977 first-network and 351 warm requests. Hidden network 93 failed at residual 0.422s, but all 50 public networks completed; the corrected grade became the new public best at -0.291% versus #336342.

Recorded measurement:

    {"score_public": 1.4759020254247964e-09, "n": 100, "failed_networks": 1, "utilization": 0.11228666127665747, "wall_mean_s": 95.683, "wall_max_s": 107.999, "resid_mean_s": 0.355, "resid_max_s": 0.422}

Public score: 1.4759020254247964e-09. Status: GRADED. Failed networks: 1.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3893](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3893)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3998](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3998)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4119](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4119)
- [SUBMISSION_LEDGER.md#L1024](../../records/SUBMISSION_LEDGER.md#L1024)
- [SUBMISSION_LEDGER.md#L1027](../../records/SUBMISSION_LEDGER.md#L1027)
- [SUBMISSION_LEDGER.md#L1028](../../records/SUBMISSION_LEDGER.md#L1028)

<details>
<summary>All result files and records (11)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ROOT_FIRE_RECEIPT_v1.json](ROOT_FIRE_RECEIPT_v1.json) · result
- [ROOT_PUBLIC_GRADE_v1.json](ROOT_PUBLIC_GRADE_v1.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [validate.json](validate.json) · record
- [stage_a.json](stage_a.json) · record
- [package.json](package.json) · record
- [validate_package.json](validate_package.json) · record
- [ROOT_PUBLIC_GRADE_v1-428ebecc.json](ROOT_PUBLIC_GRADE_v1-428ebecc.json) · result
- [eval_336412.json](eval_336412.json) · result

</details>

## Related work

- ← shipped in: [K-split key refit and FIFO-to-LRU strip caches: identical predictions, 0.03% fewer FLOPs](../../experiments/arithmetic-kernels/k-split-key-refit-fifo-lru-strip-caches-identical/README.md)
- ← shipped in: [Public leaderboard scores use the 50 public networks even when hidden networks fail](../../records/research/2026-10-16-public-leaderboard-scores-use-50-public-networks-even-when/README.md)
- ← shipped in: [Rank-384 stack census refit: 977 fewer first and 351 fewer warm requests](../../experiments/setup-budget/rank-384-stack-census-refit-977-fewer-first-351/README.md)

[Research graph](../../graph/index.html#node=submission-336412) · [Copy provenance](../../PROVENANCE.json)

