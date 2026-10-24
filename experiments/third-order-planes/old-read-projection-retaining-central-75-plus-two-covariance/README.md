# Old-read projection retaining central 75% plus two covariance/mean rows: pilot-8 adjusted +0.278695%

2026-10-15T15:08:59Z · experiment · rejected · measured

Retaining the central 75% of old rows plus two covariance/mean directions failed the fixed pilot on the QR parent. Saved outer-only fitting and row-allocation rescues recovered too little weighted error.

Recorded measurement:

    vs QR MSE +1.440453%, bill -1.161758%, adjusted +0.278695%; vs e406 adjusted +0.244585%; outer-only and rowallocation recovered <7% weighted error

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_protected_old_projection_v1/grade_retired_phase_v1.py](code/lead/runs/astra_protected_old_projection_v1/grade_retired_phase_v1.py)
- [code/lead/runs/astra_protected_old_projection_v1/build_source_and_tiny_HELD_v1.py](code/lead/runs/astra_protected_old_projection_v1/build_source_and_tiny_HELD_v1.py)
- [code/lead/runs/astra_protected_old_projection_v1/candidate_protected_old_QR_ON_HELD_v1.py](code/lead/runs/astra_protected_old_projection_v1/candidate_protected_old_QR_ON_HELD_v1.py)
- [code/lead/runs/astra_protected_old_projection_v1/fragment_protected_old_HELD_v1.py](code/lead/runs/astra_protected_old_projection_v1/fragment_protected_old_HELD_v1.py)
- [code/lead/runs/astra_protected_old_projection_v1/finite_pop_astra_v1.py](code/lead/runs/astra_protected_old_projection_v1/finite_pop_astra_v1.py)
- [code/lead/runs/astra_protected_old_projection_v1/native_pop_judge_astra_v1.py](code/lead/runs/astra_protected_old_projection_v1/native_pop_judge_astra_v1.py)
- [code/lead/runs/astra_protected_old_projection_v1/prepare_protocol_HELD_v1.py](code/lead/runs/astra_protected_old_projection_v1/prepare_protocol_HELD_v1.py)
- [code/lead/runs/astra_read_column_oracle_v1/fragment_reference_capture_HELD_v1.py](code/lead/runs/astra_read_column_oracle_v1/fragment_reference_capture_HELD_v1.py)
- [code/lead/runs/astra_read_column_oracle_v1/check_reference_capture_HELD_v1.py](code/lead/runs/astra_read_column_oracle_v1/check_reference_capture_HELD_v1.py)
- [code/lead/runs/astra_read_column_oracle_v1/inspect_protected_outer_fit_HELD_v1.py](code/lead/runs/astra_read_column_oracle_v1/inspect_protected_outer_fit_HELD_v1.py)
- [code/lead/runs/astra_read_column_oracle_v1/inspect_protected_outer_fit_HELD_v2.py](code/lead/runs/astra_read_column_oracle_v1/inspect_protected_outer_fit_HELD_v2.py)
- [code/lead/runs/astra_read_column_oracle_v1/price_band_select_HELD_v1.py](code/lead/runs/astra_read_column_oracle_v1/price_band_select_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (32)</summary>

- [code/lead/runs/astra_read_column_oracle_v1/price_fixed_minus64_HELD_v1.py](code/lead/runs/astra_read_column_oracle_v1/price_fixed_minus64_HELD_v1.py)
- [code/lead/runs/astra_read_column_oracle_v1/price_fixed_minus64_HELD_v2.py](code/lead/runs/astra_read_column_oracle_v1/price_fixed_minus64_HELD_v2.py)
- [code/lead/runs/astra_read_column_oracle_v1/price_protected_old_HELD_v1.py](code/lead/runs/astra_read_column_oracle_v1/price_protected_old_HELD_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
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
- [code/lead/runs/astra_read_column_oracle_v1/inspect_fixed_band_oracle_HELD_v1.py](code/lead/runs/astra_read_column_oracle_v1/inspect_fixed_band_oracle_HELD_v1.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py)
- [code/lead/runs/astra_rtt2_prefix_z1_late_v1/finite_pzl_astra_v1.py](code/lead/runs/astra_rtt2_prefix_z1_late_v1/finite_pzl_astra_v1.py)
- [code/lead/runs/astra_rtt2_prefix_z1_late_v1/grade_retired_phase_v1.py](code/lead/runs/astra_rtt2_prefix_z1_late_v1/grade_retired_phase_v1.py)
- [code/lead/runs/astra_rtt2_prefix_z1_late_v1/native_pzl_judge_astra_v1.py](code/lead/runs/astra_rtt2_prefix_z1_late_v1/native_pzl_judge_astra_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3398](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3398)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3400](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3400)

<details>
<summary>All result files and records (10)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [SOURCE_IMPORT_ONLY_v1.json](SOURCE_IMPORT_ONLY_v1.json) · result
- [SOURCE_READY_HELD_v1.json](SOURCE_READY_HELD_v1.json) · result
- [PILOT_RETIREMENT_astra_v1.json](PILOT_RETIREMENT_astra_v1.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result
- [TRAIN16_PROTECTED_OUTER_FIT_v1.json](TRAIN16_PROTECTED_OUTER_FIT_v1.json) · result
- [TRAIN16_PROTECTED_OUTER_FIT_v2.json](TRAIN16_PROTECTED_OUTER_FIT_v2.json) · result
- [CORE96_BAND160_SELECT32_TARIFF_v1.json](CORE96_BAND160_SELECT32_TARIFF_v1.json) · result
- [FIXED_MINUS64_TARIFF_v2.json](FIXED_MINUS64_TARIFF_v2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-old-read-projection-retaining-central-75-plus-two-covariance) · [Copy provenance](../../../PROVENANCE.json)

