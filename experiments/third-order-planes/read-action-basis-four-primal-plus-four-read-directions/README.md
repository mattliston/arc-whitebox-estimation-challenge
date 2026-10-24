# Read-action basis with four primal plus four read directions at a fixed 248+8 budget: pilot-8 adjusted +0.25%, closed

2026-10-15T02:09:46Z · experiment · rejected · measured

Spending eight of the 256 read coordinates on four primal and four read-action directions that keep the raw actions coherent (forecast +0.073 to +0.075% bill) lost accuracy on the 8-network pilot.

Recorded measurement:

    adjusted +0.2484266%, MSE +0.1743844%, bill +0.0739134%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py)
- [code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)

<details>
<summary>Additional programs and dependencies (14)</summary>

- [code/lead/runs/astra_read_action_primal4_v1/build_rap4_source_HELD_v2.py](code/lead/runs/astra_read_action_primal4_v1/build_rap4_source_HELD_v2.py)
- [code/lead/runs/astra_read_action_primal4_v1/candidate_rap4_HELD_v2.py](code/lead/runs/astra_read_action_primal4_v1/candidate_rap4_HELD_v2.py)
- [code/lead/runs/astra_read_action_primal4_v1/finite_rap4_interfaces_HELD_v1.py](code/lead/runs/astra_read_action_primal4_v1/finite_rap4_interfaces_HELD_v1.py)
- [code/lead/runs/astra_read_action_primal4_v1/finite_rap4_science_HELD_v2.py](code/lead/runs/astra_read_action_primal4_v1/finite_rap4_science_HELD_v2.py)
- [code/lead/runs/astra_read_action_primal4_v1/fragment_rap4_HELD_v2.py](code/lead/runs/astra_read_action_primal4_v1/fragment_rap4_HELD_v2.py)
- [code/lead/runs/astra_read_action_primal4_v1/kernel_rap4_HELD_v1.py](code/lead/runs/astra_read_action_primal4_v1/kernel_rap4_HELD_v1.py)
- [code/lead/runs/astra_read_action_primal4_v1/native_rap4_science_judge_HELD_v1.py](code/lead/runs/astra_read_action_primal4_v1/native_rap4_science_judge_HELD_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3266](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3266)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3275](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3275)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-read-action-basis-four-primal-plus-four-read-directions) · [Copy provenance](../../../PROVENANCE.json)

