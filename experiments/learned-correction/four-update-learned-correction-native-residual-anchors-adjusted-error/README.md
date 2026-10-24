# Four-update learned correction with native residual anchors: adjusted error rises 0.5246001601%

2026-09-30 · experiment · rejected · measured

Four Adam updates used16 native residual anchors while freezing the44144-parameter trunk and fitting3437 existing main/head-edge weights plus10 dials.

Recorded measurement:

    TRAIN2 adjusted+.5246001601%,raw+.52379822%,bill0%,1win,maxreg1.23203%;training545.452154s,6.650314752GB,zero swap.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/candidate.py)
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
<summary>Additional programs and dependencies (109)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/boundary_feasibility_engine_v1/assess.py](code/codex_resume_20260926_v2/boundary_feasibility_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/boundary_feasibility_engine_v3/assess.py](code/codex_resume_20260926_v2/boundary_feasibility_engine_v3/assess.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/handle_client_v2.py](code/codex_resume_20260926_v2/handle_client_v2.py)
- [code/codex_resume_20260926_v2/handle_server_v2.py](code/codex_resume_20260926_v2/handle_server_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_d5_late_common224_v1/build.py](code/codex_resume_20260926_v2/hill_climb_d5_late_common224_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_d5_late_common224_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_d5_late_common224_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/build.py](code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_gnbatch_young_trace_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_gnbatch_young_trace_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/build.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/probe.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_hk15_blocked_v2/kernel_blocked.py](code/codex_resume_20260926_v2/hill_climb_hk15_blocked_v2/kernel_blocked.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/build.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/probe.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/build.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/build.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/fragment.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/wrapper.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/wrapper.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/boundary_feasibility_assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/boundary_feasibility_assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/build.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/paired_boundary_assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/paired_boundary_assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess_extension.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess_extension.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/build.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/make_assessor.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/make_assessor.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/precision.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/precision.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/build_native2_config.py](code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/build_native2_config.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/export.py](code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/export.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/offline_guard.py](code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/offline_guard.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/prepare_protocol.py](code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/prepare_protocol.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/probe.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/train.py](code/codex_resume_20260926_v2/hill_climb_s34_anchored_small4_v1/train.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/build.py](code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/probe.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_young_negative_trace_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_young_negative_trace_v1/fragment.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)
- [code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py](code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py)
- [code/codex_resume_20260926_v2/official_limit_research_extension_v1/native8.py](code/codex_resume_20260926_v2/official_limit_research_extension_v1/native8.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/codex_resume_20260926_v2/run_handle_worker_v3.py](code/codex_resume_20260926_v2/run_handle_worker_v3.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/assess.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/assess.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/direction_common.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/direction_common.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/native_parameters.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/native_probe.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/native_probe.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/torch_probe.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/torch_probe.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/torch_runtime_v2.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/torch_runtime_v2.py)
- [code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/candidate.py](code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/candidate.py)
- [code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/export.py](code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/export.py)
- [code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/native_probe.py](code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/native_probe.py)
- [code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/offline_guard.py](code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/offline_guard.py)
- [code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/train.py](code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/train.py)
- [code/codex_resume_20260926_v2/serving_engine_v1/assess.py](code/codex_resume_20260926_v2/serving_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/serving_engine_v1/common.py](code/codex_resume_20260926_v2/serving_engine_v1/common.py)
- [code/codex_resume_20260926_v2/serving_engine_v1/run.py](code/codex_resume_20260926_v2/serving_engine_v1/run.py)
- [code/codex_resume_20260926_v2/serving_engine_v2/assess.py](code/codex_resume_20260926_v2/serving_engine_v2/assess.py)
- [code/codex_resume_20260926_v2/serving_engine_v2/common.py](code/codex_resume_20260926_v2/serving_engine_v2/common.py)
- [code/codex_resume_20260926_v2/serving_engine_v2/run.py](code/codex_resume_20260926_v2/serving_engine_v2/run.py)
- [code/codex_resume_20260926_v2/setup_only_client_v1.py](code/codex_resume_20260926_v2/setup_only_client_v1.py)
- [code/codex_resume_20260926_v2/setup_stress_driver_v1.py](code/codex_resume_20260926_v2/setup_stress_driver_v1.py)
- [code/codex_resume_20260926_v2/valid40_engine_v1/assess.py](code/codex_resume_20260926_v2/valid40_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/valid40_engine_v1/run.py](code/codex_resume_20260926_v2/valid40_engine_v1/run.py)
- [code/codex_resume_20260926_v2/valid40_engine_v2/assess.py](code/codex_resume_20260926_v2/valid40_engine_v2/assess.py)
- [code/codex_resume_20260926_v2/valid40_engine_v2/build_configs.py](code/codex_resume_20260926_v2/valid40_engine_v2/build_configs.py)
- [code/codex_resume_20260926_v2/valid40_engine_v2/run.py](code/codex_resume_20260926_v2/valid40_engine_v2/run.py)
- [code/codex_resume_20260926_v2/valid40_official_limit_v1/assess.py](code/codex_resume_20260926_v2/valid40_official_limit_v1/assess.py)
- [code/codex_resume_20260926_v2/valid40_official_limit_v1/build_configs.py](code/codex_resume_20260926_v2/valid40_official_limit_v1/build_configs.py)
- [code/codex_resume_20260926_v2/valid40_official_limit_v1/run.py](code/codex_resume_20260926_v2/valid40_official_limit_v1/run.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_covhunter_v1/s6bcard_joint2t.py](code/fable_covhunter_v1/s6bcard_joint2t.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/mps_card.py](code/fable_engineer_v1/mps_card.py)
- [code/fable_engineer_v1/mps_lock.py](code/fable_engineer_v1/mps_lock.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/refit_mps_v3j2.py](code/fable_physicist_v1/refit6b/refit_mps_v3j2.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [handoff_root_archive_20260930T024527Z.md#L10](../../arithmetic-kernels/depth-three-covariance-diagonal-products-adjusted-gain-0-355945/handoff_root_archive_20260930T024527Z.md#L10)

<details>
<summary>All result files and records (5)</summary>

- [handoff_root_archive_20260930T024527Z.md](../../arithmetic-kernels/depth-three-covariance-diagonal-products-adjusted-gain-0-355945/handoff_root_archive_20260930T024527Z.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-four-update-learned-correction-native-residual-anchors-adjusted-error) · [Copy provenance](../../../PROVENANCE.json)

