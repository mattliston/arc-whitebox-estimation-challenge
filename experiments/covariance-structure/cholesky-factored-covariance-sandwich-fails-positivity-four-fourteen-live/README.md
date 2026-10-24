# Cholesky-factored covariance sandwich fails positivity on four of fourteen live layers

2026-10-16T13:47:30Z · experiment · rejected · measured

The proposed triangular factorization was killed at its first pre-registered condition because live covariance matrices were numerically indefinite in both float32 and float64.

Recorded measurement:

    4/14=29%;mineigen-6.7e-8..-1.1e-6;relative-8e-9..-1.1e-7;successfulrouteerror8e-7..1e-6

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_chol_v1/pd_census_v1.py](code/lead/runs/opus_chol_v1/pd_census_v1.py)
- [code/lead/runs/opus_chol_v1/capture_cov_v1.py](code/lead/runs/opus_chol_v1/capture_cov_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4180](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4180)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4172](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4172)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [guard_cap.json](guard_cap.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-cholesky-factored-covariance-sandwich-fails-positivity-four-fourteen-live) · [Copy provenance](../../../PROVENANCE.json)

