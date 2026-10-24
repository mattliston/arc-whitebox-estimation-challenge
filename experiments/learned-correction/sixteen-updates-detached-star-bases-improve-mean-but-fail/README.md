# Sixteen updates with detached star bases improve the mean but fail both confidence bounds

2026-09-30 · experiment · rejected · measured

A fixed continuation to16 total updates preserved the eight-update checkpoint, optimizer state, random state, order and schedule. Its own eight-network native result failed the prospective confidence gates and was closed.

Recorded measurement:

    N8 raw-.3122248293%,adjusted-.4358105439%,5wins,maxreg.82465254%;CI upper raw+.17662559%,adjusted+.05578224%;finite16/L0/replay/resources passed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/train_segment.py)
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

<details>
<summary>Additional programs and dependencies (36)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/assess_validation.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/assess_validation.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/build_trainer.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/build_trainer.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/export_endpoint.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/export_endpoint.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/finalize_training.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/finalize_training.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/prepare_validation.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/prepare_validation.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/validate_endpoint.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/validate_endpoint.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/validation_common.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star16_v1/validation_common.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/export_endpoint.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/export_endpoint.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/finalize_training.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/finalize_training.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/train_segment.py)
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
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [HILL_CLIMB_CURRENT.md#L289](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L289)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [build.json](build.json) · result
- [protocol.json](protocol.json) · result
- [native_assessment.json](native_assessment.json) · result
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-sixteen-updates-detached-star-bases-improve-mean-but-fail) · [Copy provenance](../../../PROVENANCE.json)

