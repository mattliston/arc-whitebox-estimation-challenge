# Read-prefix cutoff reduced from 1.5 to 1.0 standard deviations: pilot adjusted +2.3253677656640717%

2026-10-15T14:33:11Z · experiment · rejected · measured

Changing only the active young-read prefix cutoff from 1.5 to 1.0 standard deviations on the signed-prefix power-step and QR parent failed the paired pilot. The rank-two completion, original wider mask, QR refinement and exact old tail were unchanged; no remaining sixteen or delivery evaluation ran.

Recorded measurement:

    versus QR: MSE_log_pct 4.523736689536677, bill_log_pct -2.1983689238726054, net_log_pct 2.3253677656640717, ops_delta 255.375; versus e406: MSE_log_pct 4.4486379963693, bill_log_pct -2.1573799652788272, net_log_pct 2.2912580310904724

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_rtt2_prefix_z1_v1/grade_pilot_astra_v1.py](code/lead/runs/astra_rtt2_prefix_z1_v1/grade_pilot_astra_v1.py)
- [code/lead/runs/astra_rtt2_prefix_z1_v1/build_prefix_z1_HELD_v1.py](code/lead/runs/astra_rtt2_prefix_z1_v1/build_prefix_z1_HELD_v1.py)
- [code/lead/runs/astra_rtt2_prefix_z1_v1/candidate_QR_prefix_z1_ON_HELD_v1.py](code/lead/runs/astra_rtt2_prefix_z1_v1/candidate_QR_prefix_z1_ON_HELD_v1.py)
- [code/lead/runs/astra_rtt2_prefix_z1_v1/price_prefix_z1_HELD_v1.py](code/lead/runs/astra_rtt2_prefix_z1_v1/price_prefix_z1_HELD_v1.py)
- [code/lead/runs/astra_rtt2_prefix_z1_v1/finite_pz1_astra_v1.py](code/lead/runs/astra_rtt2_prefix_z1_v1/finite_pz1_astra_v1.py)
- [code/lead/runs/astra_rtt2_prefix_z1_v1/native_pz1_judge_astra_v1.py](code/lead/runs/astra_rtt2_prefix_z1_v1/native_pz1_judge_astra_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

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
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3389](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3389)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3391](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3391)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [PREFIX_Z1_TRAIN16_TARIFF_BOUNDS_v1.json](PREFIX_Z1_TRAIN16_TARIFF_BOUNDS_v1.json) · result
- [SOURCE_READY_HELD_v1.json](SOURCE_READY_HELD_v1.json) · result
- [ACTUAL_PILOT_EXITS_astra_v1.json](ACTUAL_PILOT_EXITS_astra_v1.json) · result
- [PILOT_RETIREMENT_astra_v1.json](PILOT_RETIREMENT_astra_v1.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-read-prefix-cutoff-reduced-1-5-1-0-standard) · [Copy provenance](../../../PROVENANCE.json)

