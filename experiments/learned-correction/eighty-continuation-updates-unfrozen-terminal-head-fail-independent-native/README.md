# Eighty continuation updates with an unfrozen terminal head fail independent native confirmation

2026-09-30 · experiment · rejected · measured

A fixed80-update continuation restored the original head training and full trunk, with two passes over160 examples. Initial eight-network gains failed wins/confidence gates; one separately registered unchanged-endpoint eight-network confirmation regressed, and the combined16-network result was not retained.

Recorded measurement:

    Initial8 raw-.3415874274764641%,adjusted-.45531707436000657%,5wins,CIcross;new8 raw+.15596554128793905%,adjusted+.05096232783889931%,3adjusted/2raw wins;pooled16 raw-.10209605371050978%,adjusted-.2090638369409592%,8adjusted/7raw wins,both CIs cross0,maxreg1.01342574%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/train_segment.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/export_endpoint.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/export_endpoint.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)

<details>
<summary>Additional programs and dependencies (64)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
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
- [code/codex_resume_20260926_v2/hill_climb_rrpi_extra32_warm_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_extra32_warm_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_extra32_warm_v1/export.py](code/codex_resume_20260926_v2/hill_climb_rrpi_extra32_warm_v1/export.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_extra32_warm_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_extra32_warm_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_extra32_warm_v1/train.py](code/codex_resume_20260926_v2/hill_climb_rrpi_extra32_warm_v1/train.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/after_training.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/after_training.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/assess_confirmation.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/assess_confirmation.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/assess_resources.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/assess_resources.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/assess_validation.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/assess_validation.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/controller.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/controller.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/prepare_validation.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/prepare_validation.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/validate_endpoint.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/validate_endpoint.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/validation_common.py](code/codex_resume_20260926_v2/hill_climb_rrpi_full80_v1/validation_common.py)
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

- [HILL_CLIMB_CURRENT.md#L385](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L385)
- [HILL_CLIMB_CURRENT.md#L398](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L398)

<details>
<summary>All result files and records (7)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [build.json](build.json) · result
- [protocol.json](protocol.json) · result
- [native_assessment.json](native_assessment.json) · result
- [confirmation_assessment.json](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/confirmation_assessment.json) · result
- [resource_assessment.json](resource_assessment.json) · result
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-eighty-continuation-updates-unfrozen-terminal-head-fail-independent-native) · [Copy provenance](../../../PROVENANCE.json)

