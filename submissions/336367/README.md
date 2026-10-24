# Submission #336367: layer-10 rank 384, compact kernels and exact request cuts; seven runtime failures

2026-10-16T02:09:36Z · submission · failed · measured

The rank 384 middle join, transport-kurtosis strength 0.6, expanded grouped reads, lazy compact layout and exact view/tree-load request cuts passed local gates. Seven time/residual failures invalidated the public result, although 45 completed public networks measured score-aggregate -0.208% versus #336342.

Recorded measurement:

    {"score_public": 0.09688601148991585, "n": 100, "failed_networks": 7, "utilization": 0.1124221277547349, "wall_mean_s": 97.681, "wall_max_s": 112.352, "resid_mean_s": 0.362, "resid_max_s": 0.455}

Public score: 0.09688601148991585. Status: GRADED. Failed networks: 7.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3683](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3683)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3742](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3742)
- [SUBMISSION_LEDGER.md#L1009](../../records/SUBMISSION_LEDGER.md#L1009)
- [SUBMISSION_LEDGER.md#L1011](../../records/SUBMISSION_LEDGER.md#L1011)

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
- [eval_336367.json](eval_336367.json) · result

</details>

## Related work

- ← shipped in: [Compact-layout frames bound lazily: slow setup 4.611 to 4.585 seconds for 809,972,800 first-network FLOPs](../../experiments/setup-budget/compact-layout-frames-bound-lazily-slow-setup-4-611/README.md)
- ← shipped in: [Exact repeated-view cache: 179 fewer first-network and 199 fewer warm requests](../../experiments/wall-and-residual/exact-repeated-view-cache-179-fewer-first-network-199/README.md)
- ← shipped in: [Layer-10 join rank 384: 24-network noise-free adjusted -0.461%, final eight +0.173%](../../experiments/join-sketch/layer-10-join-rank-384-24-network-noise-free/README.md)
- ← shipped in: [Rank-384 join plus kurtosis strength 0.6: fresh-24 noise-free adjusted -0.199%, aggregate -0.096%](../../experiments/join-sketch/rank-384-join-plus-kurtosis-strength-0-6-fresh/README.md)
- ← shipped in: [Transport-gate kurtosis strength 0.6: fresh-eight noise-free MSE -0.072% at identical FLOPs](../../experiments/dials/transport-gate-kurtosis-strength-0-6-fresh-eight-noise/README.md)
- ← shipped in: [Tree loads replaced by two unstacks and a cached half view: 465/506 fewer requests at identical FLOPs](../../experiments/arithmetic-kernels/tree-loads-replaced-two-unstacks-cached-half-view-465/README.md)

[Research graph](../../graph/index.html#node=submission-336367) · [Copy provenance](../../PROVENANCE.json)

