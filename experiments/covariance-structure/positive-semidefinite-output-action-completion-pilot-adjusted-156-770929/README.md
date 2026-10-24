# Positive-semidefinite output-action completion: pilot adjusted +156.770929%

2026-10-15T21:44:27Z · experiment · rejected · measured

Replacing the completion with a paid-action Nystrom form repaired saved negative eigenvalues while retaining active blocks and summary actions, but did not fix the final-accuracy failure.

Recorded measurement:

    saved eigenmin about-.47/-.42 repaired positive;nextvarianceproxy4-6%better;pilot MSE+157.484639%,bill-.713710%,net+156.770929%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_output_action_psd_science_v1/grade_oap_retired_phase_v1.py](code/lead/runs/astra_output_action_psd_science_v1/grade_oap_retired_phase_v1.py)
- [code/lead/runs/astra_output_action_psd_science_v1/oap_source_contract_v1.py](code/lead/runs/astra_output_action_psd_science_v1/oap_source_contract_v1.py)
- [code/lead/runs/astra_output_action_psd_science_v1/finite_oap_astra_v1.py](code/lead/runs/astra_output_action_psd_science_v1/finite_oap_astra_v1.py)
- [code/lead/runs/astra_output_action_psd_science_v1/native_oap_judge_astra_v1.py](code/lead/runs/astra_output_action_psd_science_v1/native_oap_judge_astra_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
- [code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (14)</summary>

- [code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3543](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3543)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3541](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3541)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [PILOT_RETIREMENT_astra_v1.json](PILOT_RETIREMENT_astra_v1.json) · result
- [PREPARED_FOR_ROOT_REVIEW_v1.json](PREPARED_FOR_ROOT_REVIEW_v1.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result
- [ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json](ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-positive-semidefinite-output-action-completion-pilot-adjusted-156-770929) · [Copy provenance](../../../PROVENANCE.json)

