# Submission #336351: expanded 167-shape read dispatch and first-read reallocation; three runtime failures

2026-10-16T00:31:03Z · submission · failed · measured

Expanding the grouped read table to 167 shapes and reallocating the first read passed local gates but failed two 120-second walls and one residual limit on platform. Kernel slowdown was +4.4% mean and up to+22% on matched networks; the public score included zero-prediction fallbacks.

Recorded measurement:

    {"score_public": 0.05669618864209241, "n": 100, "failed_networks": 3, "utilization": 0.11034408905057717, "wall_mean_s": 95.959, "wall_max_s": 118.296, "resid_mean_s": 0.35, "resid_max_s": 0.484}

Public score: 0.05669618864209241. Status: GRADED. Failed networks: 3.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3597](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3597)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3625](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3625)
- [SUBMISSION_LEDGER.md#L1005](../../records/SUBMISSION_LEDGER.md#L1005)
- [SUBMISSION_LEDGER.md#L1007](../../records/SUBMISSION_LEDGER.md#L1007)

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
- [eval_336351.json](eval_336351.json) · result

</details>

## Related work

- ← shipped in: [Kernel dispatch expanded to 167 shapes with layer-one read reallocation: all-24 adjusted -0.2096793%](../../experiments/arithmetic-kernels/kernel-dispatch-expanded-167-shapes-layer-one-read-reallocation/README.md)

[Research graph](../../graph/index.html#node=submission-336351) · [Copy provenance](../../PROVENANCE.json)

