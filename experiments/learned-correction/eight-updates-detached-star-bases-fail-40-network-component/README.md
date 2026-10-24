# Eight updates with detached star bases fail 40-network component confirmation

2026-09-30 · experiment · rejected · measured

The fixed eight-update endpoint improved mean error on a small panel but failed its initial maximum-regression gate and later confidence gates. A separate prospective full40 component study also failed, so the endpoint was not retained.

Recorded measurement:

    Initial2 raw-.1140296804%,adjusted-.2546237223%,1win,maxreg1.08022178%;pooled8 raw-.384985%,adjusted-.502577%,6wins,CIcross;full40 raw+.05290838314%,adjusted-.05472743514%,23adjusted/19raw wins,adjusted CI[-.25751905,+.15090311]%,maxreg1.08777732%;bill-.10442816% inherited.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/candidate.py)
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
<summary>Additional programs and dependencies (69)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/boundary_assess_s34.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/boundary_assess_s34.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
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
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/screen_v2.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/screen_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/valid40_engine_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/valid40_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/valid40_engine_v1/run.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/valid40_engine_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/prepare_full40.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/prepare_full40.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/valid40_engine_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/valid40_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/valid40_engine_v1/run.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/valid40_engine_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/assess_resources.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/assess_resources.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/prepare_full40.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/prepare_full40.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/valid40_engine_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/valid40_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/valid40_engine_v1/run.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/valid40_engine_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/assess_followup8.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/assess_followup8.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/assess_validation.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/assess_validation.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/export_endpoint.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/export_endpoint.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/finalize_training.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/finalize_training.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/prepare_validation.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/prepare_validation.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/validate_endpoint.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/validate_endpoint.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/validation_common.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/validation_common.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)
- [code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py](code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
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

- [HILL_CLIMB_CURRENT.md#L267](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L267)
- [HILL_CLIMB_CURRENT.md#L311](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L311)
- [HILL_CLIMB_CURRENT.md#L304](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L304)

<details>
<summary>All result files and records (7)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [build.json](build.json) · result
- [protocol.json](protocol.json) · result
- [broad_assessment.json](broad_assessment.json) · result
- [native_assessment.json](native_assessment.json) · result
- [followup_assessment.json](followup_assessment.json) · result
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-eight-updates-detached-star-bases-fail-40-network-component) · [Copy provenance](../../../PROVENANCE.json)

