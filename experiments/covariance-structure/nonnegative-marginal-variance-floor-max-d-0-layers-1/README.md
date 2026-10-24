# Nonnegative marginal variance floor max(d,0) at layers 1-14: bill +0.00005%, fresh-16 log net -0.0083%, never confirmed

2026-10-14T17:09:11Z · experiment · inconclusive · measured

A fixed mean-preserving O(n) projection that floors the calibrated float64 diagonal at zero before the paired diagonal writes (means and same-input off-diagonals unchanged) was qualified at effectively zero bill; its 16-network fresh screen passed by a hair and the out-of-sample confirmation was deferred and never run.

Recorded measurement:

    geometric bill +0.000048892807%, no accuracy selection; fresh 16: log net -0.0083057%; OOS deferred behind larger-shot screening and its OOS seeds later reassigned to the GOP2 fresh screen

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py](code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py)
- [code/lead/runs/astra_nonnegative_variance_v1/build_nnv_native_v1.py](code/lead/runs/astra_nonnegative_variance_v1/build_nnv_native_v1.py)
- [code/lead/runs/astra_nonnegative_variance_v1/current_OFF_UNVALIDATED_astra_v1.py](code/lead/runs/astra_nonnegative_variance_v1/current_OFF_UNVALIDATED_astra_v1.py)
- [code/lead/runs/astra_nonnegative_variance_v1/current_ON_UNVALIDATED_astra_v1.py](code/lead/runs/astra_nonnegative_variance_v1/current_ON_UNVALIDATED_astra_v1.py)
- [code/lead/runs/astra_nonnegative_variance_v1/finite_nnv_qualification_HELD_v1.py](code/lead/runs/astra_nonnegative_variance_v1/finite_nnv_qualification_HELD_v1.py)
- [code/lead/runs/astra_nonnegative_variance_v1/finite_nnv_science_HELD_v1.py](code/lead/runs/astra_nonnegative_variance_v1/finite_nnv_science_HELD_v1.py)
- [code/lead/runs/astra_nonnegative_variance_v1/kernel_HELD_v1.py](code/lead/runs/astra_nonnegative_variance_v1/kernel_HELD_v1.py)
- [code/lead/runs/astra_nonnegative_variance_v1/native_nnv_qualification_judge_v1.py](code/lead/runs/astra_nonnegative_variance_v1/native_nnv_qualification_judge_v1.py)
- [code/lead/runs/astra_nonnegative_variance_v1/native_nnv_science_judge_HELD_v1.py](code/lead/runs/astra_nonnegative_variance_v1/native_nnv_science_judge_HELD_v1.py)
- [code/lead/runs/astra_nonnegative_variance_v1/owned_native_runner_v1.py](code/lead/runs/astra_nonnegative_variance_v1/owned_native_runner_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)

<details>
<summary>Additional programs and dependencies (4)</summary>

- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/k4liab_s51k2tw/candidate.py](code/scout_native/k4liab_s51k2tw/candidate.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2901](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2901)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2902](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2902)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2904](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2904)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2906](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2906)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2932](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2932)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3016](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3016)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Cauchy-Schwarz off-diagonal covariance projection fails at its first on-call: the calibrated diagonal is not nonnegative](../cauchy-schwarz-off-diagonal-covariance-projection-fails-first-call/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-nonnegative-marginal-variance-floor-max-d-0-layers-1) · [Copy provenance](../../../PROVENANCE.json)

