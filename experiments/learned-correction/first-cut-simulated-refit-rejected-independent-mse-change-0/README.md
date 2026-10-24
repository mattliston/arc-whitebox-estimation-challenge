# First cut-simulated refit rejected: independent MSE change -0.235%, standard error 0.340

2026-10-11T03:27:57Z · experiment · rejected · measured

Twelve arms sized on 16 networks gave a model forecast of -0.99% and a leave-one-out of -0.40%, but the independent check read -0.235% (se 0.340): overfit, not a wrong direction; the fixed rule (check < -SE) rejected it and nothing was exported.

Recorded measurement:

    sizing 136-151 -0.912% (se 0.346, 10/16); check 152-167 -0.235% (se 0.340, 10/16); 352 requests, 96 public gradients, 12 arms

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix24.sh](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix24.sh)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)

<details>
<summary>Additional programs and dependencies (96)</summary>

- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_covhunter_v1/s6bcard_joint2t.py](code/fable_covhunter_v1/s6bcard_joint2t.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/engineer/a10_bd32/fragment_bd32.py](code/opus_reexam_20261001_v1/engineer/a10_bd32/fragment_bd32.py)
- [code/opus_reexam_20261001_v1/engineer/a11_bare_views/analyze_binds.py](code/opus_reexam_20261001_v1/engineer/a11_bare_views/analyze_binds.py)
- [code/opus_reexam_20261001_v1/engineer/a11_bare_views/fragment_bvp.py](code/opus_reexam_20261001_v1/engineer/a11_bare_views/fragment_bvp.py)
- [code/opus_reexam_20261001_v1/engineer/a12_live_slice/fragment_dcr.py](code/opus_reexam_20261001_v1/engineer/a12_live_slice/fragment_dcr.py)
- [code/opus_reexam_20261001_v1/engineer/a17_tanh/fragment_ets.py](code/opus_reexam_20261001_v1/engineer/a17_tanh/fragment_ets.py)
- [code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut.py](code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut.py)
- [code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut_v2.py](code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3.py](code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3.py)
- [code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3o.py](code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3o.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/engineer/a29_e2/fragment_e2_v1.py](code/opus_reexam_20261001_v1/engineer/a29_e2/fragment_e2_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a31_star/fragment_sx_v1.py](code/opus_reexam_20261001_v1/engineer/a31_star/fragment_sx_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a32_c2k/fragment_c2k_v1.py](code/opus_reexam_20261001_v1/engineer/a32_c2k/fragment_c2k_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/fragment_ss_v3.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/fragment_ss_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/engineer/a35_join/fragment_jr_v1.py](code/opus_reexam_20261001_v1/engineer/a35_join/fragment_jr_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a40_setup/fragment_cs_v1.py](code/opus_reexam_20261001_v1/engineer/a40_setup/fragment_cs_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bk3_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bk3_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bsk_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bsk_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_gsk_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_gsk_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v2.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v1.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v2.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v1.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v3.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a43_readout/fragment_ro_v1.py](code/opus_reexam_20261001_v1/engineer/a43_readout/fragment_ro_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v1.py](code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v2.py](code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v3.py](code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a45_l15/fragment_l15_v1.py](code/opus_reexam_20261001_v1/engineer/a45_l15/fragment_l15_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a46_b6/fragment_b6_v1.py](code/opus_reexam_20261001_v1/engineer/a46_b6/fragment_b6_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a46_b6/fragment_b6_v2.py](code/opus_reexam_20261001_v1/engineer/a46_b6/fragment_b6_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a47_kc/fragment_kc_v1.py](code/opus_reexam_20261001_v1/engineer/a47_kc/fragment_kc_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a49_band/fragment_band_v1.py](code/opus_reexam_20261001_v1/engineer/a49_band/fragment_band_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a50_b7/fragment_b7_v2.py](code/opus_reexam_20261001_v1/engineer/a50_b7/fragment_b7_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v2.py](code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a55_u15/fragment_u15_v1.py](code/opus_reexam_20261001_v1/engineer/a55_u15/fragment_u15_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a56_edge_band/edband_build_v3.py](code/opus_reexam_20261001_v1/engineer/a56_edge_band/edband_build_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a56_edge_band/fragment_edband_v1.py](code/opus_reexam_20261001_v1/engineer/a56_edge_band/fragment_edband_v1.py)
- [code/opus_reexam_20261001_v1/engineer/buildlib.py](code/opus_reexam_20261001_v1/engineer/buildlib.py)
- [code/opus_reexam_20261001_v1/engineer/edits_lib.py](code/opus_reexam_20261001_v1/engineer/edits_lib.py)
- [code/opus_reexam_20261001_v1/engineer/hzs/fragment_gen_v7.py](code/opus_reexam_20261001_v1/engineer/hzs/fragment_gen_v7.py)
- [code/opus_reexam_20261001_v1/engineer/stack/build_edband_v3.py](code/opus_reexam_20261001_v1/engineer/stack/build_edband_v3.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/trainer/cert/export_firstpoint_v1.py](code/opus_reexam_20261001_v1/trainer/cert/export_firstpoint_v1.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_protocol_v5.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_protocol_v5.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_protocol_v6.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_protocol_v6.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_v4.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_v4.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_protocol_v1.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_protocol_v1.py)
- [code/opus_reexam_20261001_v1/trainer/edband_qualification_v6.py](code/opus_reexam_20261001_v1/trainer/edband_qualification_v6.py)
- [code/opus_reexam_20261001_v1/trainer/edband_requests_v1.py](code/opus_reexam_20261001_v1/trainer/edband_requests_v1.py)
- [code/opus_reexam_20261001_v1/trainer/edband_six_arms_v3.py](code/opus_reexam_20261001_v1/trainer/edband_six_arms_v3.py)
- [code/opus_reexam_20261001_v1/trainer/edband_six_driver_v6.py](code/opus_reexam_20261001_v1/trainer/edband_six_driver_v6.py)
- [code/opus_reexam_20261001_v1/trainer/edband_six_fit_v3.py](code/opus_reexam_20261001_v1/trainer/edband_six_fit_v3.py)
- [code/opus_reexam_20261001_v1/trainer/export.py](code/opus_reexam_20261001_v1/trainer/export.py)
- [code/opus_reexam_20261001_v1/trainer/export_dials_s46_v2.py](code/opus_reexam_20261001_v1/trainer/export_dials_s46_v2.py)
- [code/opus_reexam_20261001_v1/trainer/export_edband_s46_v2.py](code/opus_reexam_20261001_v1/trainer/export_edband_s46_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_edband_v1.py](code/opus_reexam_20261001_v1/trainer/otr_edband_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)
- [code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py](code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py)
- [code/opus_reexam_20261001_v1/trainer/worker_edband_v1.py](code/opus_reexam_20261001_v1/trainer/worker_edband_v1.py)
- [code/opus_reexam_20261001_v1/trainer/worker_edband_v2.py](code/opus_reexam_20261001_v1/trainer/worker_edband_v2.py)
- [code/opus_reexam_20261001_v1/trainer/worker_generated_v3.py](code/opus_reexam_20261001_v1/trainer/worker_generated_v3.py)
- [code/opus_reexam_20261001_v1/trainer/worker_generated_v4.py](code/opus_reexam_20261001_v1/trainer/worker_generated_v4.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L811](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L811)
- [TEAM_LOG_2026-10-01_to_10-22.md#L837](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L837)
- [TEAM_LOG_2026-10-01_to_10-22.md#L954](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L954)
- [TEAM_LOG_2026-10-01_to_10-22.md#L984](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L984)
- [TEAM_LOG_2026-10-01_to_10-22.md#L987](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L987)
- [TEAM_LOG_2026-10-01_to_10-22.md#L988](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L988)
- [TEAM_LOG_2026-10-01_to_10-22.md#L990](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L990)

<details>
<summary>All result files and records (6)</summary>

- [jmNc_first_point.json](jmNc_first_point.json) · record
- [arms.json](arms.json) · record
- [astra_cut_first96_v1.txt](astra_cut_first96_v1.txt) · record
- [FINAL_RECEIPT_v1.json](FINAL_RECEIPT_v1.json) · record
- [OPUS_HANDOFF_20261011T0328Z_v1.md](OPUS_HANDOFF_20261011T0328Z_v1.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Cut-simulated research estimator removes 95% of the native prediction discrepancy](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/README.md)
- enables: [Ridge re-sizing on 32 networks adopted: pooled independent 64-network MSE change -0.323%](../../../records/research/2026-10-11-ridge-re-sizing-32-networks-adopted-pooled-independent-64/README.md)
- ← enables: [Sealed 32-network promotion protocol requires a predeclared bootstrap bound and native parity](../../../records/research/2026-10-11-sealed-32-network-promotion-protocol-requires-predeclared-bootstrap-bound/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-first-cut-simulated-refit-rejected-independent-mse-change-0) · [Copy provenance](../../../PROVENANCE.json)

