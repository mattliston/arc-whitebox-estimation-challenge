# Signed-prefix power step and QR on the completed young read: all-24 adjusted -0.069%, held for stacking

2026-10-15T05:38:06Z · experiment · adopted · measured

A signed-prefix power iteration followed by a QR refines the truncated young-read prefix of the tail-completion file (forecast +104M FLOPs, -22 charged ops). It passed the pilot and the full 24 against both the fired file and its parent, by a margin small against the uncertainty, and was held as a stacking component; it entered #336294 as the QR part of the Hermite-fusion stack.

Recorded measurement:

    pilot 8 vs e406: adjusted -0.034109735% (SE 0.214280880%, 4/8 lower), MSE -0.075098693%, bill +0.040988959%; vs d401 -0.489986373% (SE 0.372677559%, 6/8); FINAL 24 vs e406: all 24 -0.069199707% (SE 0.098331893%), OOS 8 -0.007444739% (SE 0.169515516%); vs d401 all 24 -0.592482611%, OOS 8 -0.300848794%; charged ops +7..13 vs e406

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_rtt2_prefix_power_source_v1/build_prefix_power_astra_v1.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/build_prefix_power_astra_v1.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_ON_HELD_v1.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_ON_HELD_v1.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/build_QR_runner_astra_v3.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/build_QR_runner_astra_v3.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
- [code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (31)</summary>

- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_read_tail_rank2_native_v1/candidate_rtt2_HELD_v1.py](code/lead/runs/astra_read_tail_rank2_native_v1/candidate_rtt2_HELD_v1.py)
- [code/lead/runs/astra_read_tail_rank2_native_v1/check_rtt2_v1.py](code/lead/runs/astra_read_tail_rank2_native_v1/check_rtt2_v1.py)
- [code/lead/runs/astra_read_tail_rank2_native_v1/fragment_rtt2_HELD_v1.py](code/lead/runs/astra_read_tail_rank2_native_v1/fragment_rtt2_HELD_v1.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/accept_pilot_retirement_HELD_v2.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/accept_pilot_retirement_HELD_v2.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/accept_pilot_retirement_QR_HELD_v3.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/accept_pilot_retirement_QR_HELD_v3.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_OFF_HELD_v1.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_OFF_HELD_v1.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_OFF_HELD_v2.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_OFF_HELD_v2.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v2.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v2.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/check_repaired_QR_astra_v1.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/check_repaired_QR_astra_v1.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/finite_rtp_science_HELD_v2.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/finite_rtp_science_HELD_v2.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/finite_rtp_science_HELD_v3.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/finite_rtp_science_HELD_v3.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/fragment_prefix_power_HELD_v1.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/fragment_prefix_power_HELD_v1.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/fragment_prefix_power_QR_HELD_v2.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/fragment_prefix_power_QR_HELD_v2.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/native_rtp_science_judge_HELD_v1.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/native_rtp_science_judge_HELD_v1.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/native_rtp_science_judge_HELD_v3.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/native_rtp_science_judge_HELD_v3.py)
- [code/lead/runs/astra_rtt2_request_repair_source_v1/candidate_rtt2_request_repair_HELD_v2.py](code/lead/runs/astra_rtt2_request_repair_source_v1/candidate_rtt2_request_repair_HELD_v2.py)
- [code/lead/runs/astra_rtt2_request_repair_source_v1/fragment_basis_norm_stack_HELD_v2.py](code/lead/runs/astra_rtt2_request_repair_source_v1/fragment_basis_norm_stack_HELD_v2.py)
- [code/lead/runs/astra_rtt2_request_repair_source_v1/fragment_rtt2_request_repair_HELD_v1.py](code/lead/runs/astra_rtt2_request_repair_source_v1/fragment_rtt2_request_repair_HELD_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3339](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3339)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3341](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3341)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3343](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3343)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3356](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3356)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3360](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3360)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3364](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3364)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #336294: Hermite covariance grouping and QR read with exact view caching](../../../submissions/336294/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-signed-prefix-power-step-qr-completed-young-read-all) · [Copy provenance](../../../PROVENANCE.json)

