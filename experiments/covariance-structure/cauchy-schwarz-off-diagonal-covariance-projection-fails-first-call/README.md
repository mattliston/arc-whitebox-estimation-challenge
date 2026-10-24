# Cauchy-Schwarz off-diagonal covariance projection fails at its first on-call: the calibrated diagonal is not nonnegative

2026-10-14T15:26:12Z · experiment · failed · measured

A conservative off-diagonal Cauchy-Schwarz projection at layers 1 to 14 required a preserved diagonal; the first on-call found that the original calibrated float64 diagonal already contains negative variances, violating the precondition, so the arm was closed and a nonnegative-variance repair was opened instead.

Recorded measurement:

    FAILED on first ON 7100136: finite original diagonal contains negative variance

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py](code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py)
- [code/lead/runs/astra_cauchy_covariance_projection_v1/build_cauchy_native_v1.py](code/lead/runs/astra_cauchy_covariance_projection_v1/build_cauchy_native_v1.py)
- [code/lead/runs/astra_cauchy_covariance_projection_v1/current_OFF_UNVALIDATED_astra_v1.py](code/lead/runs/astra_cauchy_covariance_projection_v1/current_OFF_UNVALIDATED_astra_v1.py)
- [code/lead/runs/astra_cauchy_covariance_projection_v1/current_ON_UNVALIDATED_astra_v1.py](code/lead/runs/astra_cauchy_covariance_projection_v1/current_ON_UNVALIDATED_astra_v1.py)
- [code/lead/runs/astra_cauchy_covariance_projection_v1/finite_csb_qualification_HELD_v1.py](code/lead/runs/astra_cauchy_covariance_projection_v1/finite_csb_qualification_HELD_v1.py)
- [code/lead/runs/astra_cauchy_covariance_projection_v1/kernel_HELD_v1.py](code/lead/runs/astra_cauchy_covariance_projection_v1/kernel_HELD_v1.py)
- [code/lead/runs/astra_cauchy_covariance_projection_v1/native_csb_qualification_judge_v1.py](code/lead/runs/astra_cauchy_covariance_projection_v1/native_csb_qualification_judge_v1.py)
- [code/lead/runs/astra_cauchy_covariance_projection_v1/owned_native_runner_v1.py](code/lead/runs/astra_cauchy_covariance_projection_v1/owned_native_runner_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/scout_native/k4liab_s51k2tw/candidate.py](code/scout_native/k4liab_s51k2tw/candidate.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2888](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2888)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2892](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2892)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Nonnegative marginal variance floor max(d,0) at layers 1-14: bill +0.00005%, fresh-16 log net -0.0083%, never confirmed](../nonnegative-marginal-variance-floor-max-d-0-layers-1/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-cauchy-schwarz-off-diagonal-covariance-projection-fails-first-call) · [Copy provenance](../../../PROVENANCE.json)

