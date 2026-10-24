# Raw-second off-diagonal feasibility projection C_ij >= -m_i m_j, layers 1-14: bill +0.046%, fresh net +0.034%, rejected

2026-10-14T15:40:07Z · experiment · rejected · measured

A fixed unit-strength projection that raises each off-diagonal entry of the float32 covariance to the raw-second feasibility bound set by the future means (mask where C32_ij < -(m+dm)_i (m+dm)_j), applied after the RC/KC compensation and before the paired capture, cost bill and did not help on the fresh screen.

Recorded measurement:

    price: full bill max log cost +0.045546388% (geomean +0.044550401%); fresh 16 net +0.033761605% log, FAIL, no OOS

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py](code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py)
- [code/lead/runs/astra_raw_second_feasibility_v1/build_raw_second_native_v1.py](code/lead/runs/astra_raw_second_feasibility_v1/build_raw_second_native_v1.py)
- [code/lead/runs/astra_raw_second_feasibility_v1/current_OFF_UNVALIDATED_astra_v1.py](code/lead/runs/astra_raw_second_feasibility_v1/current_OFF_UNVALIDATED_astra_v1.py)
- [code/lead/runs/astra_raw_second_feasibility_v1/current_ON_UNVALIDATED_astra_v1.py](code/lead/runs/astra_raw_second_feasibility_v1/current_ON_UNVALIDATED_astra_v1.py)
- [code/lead/runs/astra_raw_second_feasibility_v1/finite_r2f_qualification_HELD_v1.py](code/lead/runs/astra_raw_second_feasibility_v1/finite_r2f_qualification_HELD_v1.py)
- [code/lead/runs/astra_raw_second_feasibility_v1/finite_r2f_science_HELD_v2.py](code/lead/runs/astra_raw_second_feasibility_v1/finite_r2f_science_HELD_v2.py)
- [code/lead/runs/astra_raw_second_feasibility_v1/kernel_UNVALIDATED_v2.py](code/lead/runs/astra_raw_second_feasibility_v1/kernel_UNVALIDATED_v2.py)
- [code/lead/runs/astra_raw_second_feasibility_v1/native_r2f_qualification_judge_v1.py](code/lead/runs/astra_raw_second_feasibility_v1/native_r2f_qualification_judge_v1.py)
- [code/lead/runs/astra_raw_second_feasibility_v1/native_r2f_science_judge_HELD_v1.py](code/lead/runs/astra_raw_second_feasibility_v1/native_r2f_science_judge_HELD_v1.py)
- [code/lead/runs/astra_raw_second_feasibility_v1/owned_native_runner_v1.py](code/lead/runs/astra_raw_second_feasibility_v1/owned_native_runner_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)

<details>
<summary>Additional programs and dependencies (4)</summary>

- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/k4liab_s51k2tw/candidate.py](code/scout_native/k4liab_s51k2tw/candidate.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2870](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2870)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2874](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2874)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2880](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2880)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2894](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2894)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2898](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2898)

<details>
<summary>All result files and records (2)</summary>

- [ASTRA_FRESH_RETIREMENT_v1.json](ASTRA_FRESH_RETIREMENT_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-raw-second-off-diagonal-feasibility-projection-c-ij-m) · [Copy provenance](../../../PROVENANCE.json)

