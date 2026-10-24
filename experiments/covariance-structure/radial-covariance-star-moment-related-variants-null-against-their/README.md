# Radial covariance, star-moment and related variants are null against their predeclared gain thresholds

2026-10-13T04:41Z · experiment · rejected · measured

Besides the native algebra stack, the independent hunt's measured physics items all read null and were closed: a radial covariance model (RSCOV), a star-moment form of the newborn beta (J3), an item Sq, and ablations of the algebra stack (no-D96 / core / canonical read-leaf fusion) that did not improve on it. A common-factor covariance model (CFCOV, metered FLOPs +1.6..2.4% by source estimate against a -2% screen bar), an anisotropic diagnostic, a beta residual helper and the rounding-conditioning variants were parked unrun at the hand-off.

Recorded measurement:

    RSCOV1 / RSCOV8 radial covariance mean log -0.017% (se 0.034; -0.0167023%, SE 0.0342610) vs a -1% gate; star-moment beta J3 -0.03% (se 0.16); Sq +0.016%; QZ7 no-op; CFCOV parked at bill +1.6..2.4% estimated

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_hunt_20261013_v1/bank_algebra/fragment_read_leaf_fuse_v1.py](code/astra_hunt_20261013_v1/bank_algebra/fragment_read_leaf_fuse_v1.py)
- [code/astra_hunt_20261013_v1/beta_rule/astra_runtime_safety_v1.py](code/astra_hunt_20261013_v1/beta_rule/astra_runtime_safety_v1.py)
- [code/astra_hunt_20261013_v1/beta_rule/compare_starmoment_v2.py](code/astra_hunt_20261013_v1/beta_rule/compare_starmoment_v2.py)
- [code/astra_hunt_20261013_v1/beta_rule/cutsim_starmoment_v2/n6y_cutsim_v2.py](code/astra_hunt_20261013_v1/beta_rule/cutsim_starmoment_v2/n6y_cutsim_v2.py)
- [code/astra_hunt_20261013_v1/beta_rule/cutsim_starmoment_v2/pbase_cutsim_v1.py](code/astra_hunt_20261013_v1/beta_rule/cutsim_starmoment_v2/pbase_cutsim_v1.py)
- [code/astra_hunt_20261013_v1/beta_rule/cutsim_starmoment_v2/torch_cutsim_s46_v11.py](code/astra_hunt_20261013_v1/beta_rule/cutsim_starmoment_v2/torch_cutsim_s46_v11.py)
- [code/astra_hunt_20261013_v1/beta_rule/cutsim_starmoment_v2/torch_push_native_v2.py](code/astra_hunt_20261013_v1/beta_rule/cutsim_starmoment_v2/torch_push_native_v2.py)
- [code/astra_hunt_20261013_v1/beta_rule/cutsim_starmoment_v2/torch_rowcuts_v15.py](code/astra_hunt_20261013_v1/beta_rule/cutsim_starmoment_v2/torch_rowcuts_v15.py)
- [code/astra_hunt_20261013_v1/beta_rule/launch_starmoment_v2.sh](code/astra_hunt_20261013_v1/beta_rule/launch_starmoment_v2.sh)
- [code/astra_hunt_20261013_v1/beta_rule/otr_cut_runtime_astra_v1.py](code/astra_hunt_20261013_v1/beta_rule/otr_cut_runtime_astra_v1.py)
- [code/astra_hunt_20261013_v1/beta_rule/otr_m_starmoment_v1.py](code/astra_hunt_20261013_v1/beta_rule/otr_m_starmoment_v1.py)
- [code/astra_hunt_20261013_v1/beta_rule/prepare_starmoment_layout_v2.py](code/astra_hunt_20261013_v1/beta_rule/prepare_starmoment_layout_v2.py)

<details>
<summary>Additional programs and dependencies (86)</summary>

- [code/astra_hunt_20261013_v1/beta_rule/screen_entry_starmoment_v1.py](code/astra_hunt_20261013_v1/beta_rule/screen_entry_starmoment_v1.py)
- [code/astra_hunt_20261013_v1/beta_rule/worker_starmoment_v1.py](code/astra_hunt_20261013_v1/beta_rule/worker_starmoment_v1.py)
- [code/astra_hunt_20261013_v1/compare_cert.py](code/astra_hunt_20261013_v1/compare_cert.py)
- [code/astra_hunt_20261013_v1/guarded_entry.py](code/astra_hunt_20261013_v1/guarded_entry.py)
- [code/astra_hunt_20261013_v1/harness_audit/astra_runtime_safety_v1.py](code/astra_hunt_20261013_v1/harness_audit/astra_runtime_safety_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/compare_rscov_additive_arm_v1.py](code/astra_hunt_20261013_v1/harness_audit/compare_rscov_additive_arm_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/compare_sq_v1.py](code/astra_hunt_20261013_v1/harness_audit/compare_sq_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/launch_rscov_additive_arm_v1.sh](code/astra_hunt_20261013_v1/harness_audit/launch_rscov_additive_arm_v1.sh)
- [code/astra_hunt_20261013_v1/harness_audit/launch_sqarm_snapshot_v1.sh](code/astra_hunt_20261013_v1/harness_audit/launch_sqarm_snapshot_v1.sh)
- [code/astra_hunt_20261013_v1/harness_audit/otr_cut_runtime_astra_v1.py](code/astra_hunt_20261013_v1/harness_audit/otr_cut_runtime_astra_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/postcheck_rscov_v1.py](code/astra_hunt_20261013_v1/harness_audit/postcheck_rscov_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/prepare_rscov_additive_arm_v1.py](code/astra_hunt_20261013_v1/harness_audit/prepare_rscov_additive_arm_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/prepare_rscov_v1.py](code/astra_hunt_20261013_v1/harness_audit/prepare_rscov_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/rscov_additive_gate_v1.py](code/astra_hunt_20261013_v1/harness_audit/rscov_additive_gate_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/screen_entry_rscov_additive_arm_v1.py](code/astra_hunt_20261013_v1/harness_audit/screen_entry_rscov_additive_arm_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/screen_entry_v2.py](code/astra_hunt_20261013_v1/harness_audit/screen_entry_v2.py)
- [code/astra_hunt_20261013_v1/harness_audit/sq_arm_entry_v1.py](code/astra_hunt_20261013_v1/harness_audit/sq_arm_entry_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/worker_astra_v1.py](code/astra_hunt_20261013_v1/harness_audit/worker_astra_v1.py)
- [code/astra_hunt_20261013_v1/native_core_v1/candidate.py](code/astra_hunt_20261013_v1/native_core_v1/candidate.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_rscov_v1/n6y_cutsim_v2.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_rscov_v1/n6y_cutsim_v2.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_rscov_v1/pbase_cutsim_v1.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_rscov_v1/pbase_cutsim_v1.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_rscov_v1/torch_cutsim_s46_v11.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_rscov_v1/torch_cutsim_s46_v11.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_rscov_v1/torch_push_native_v2.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_rscov_v1/torch_push_native_v2.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_rscov_v1/torch_rowcuts_v15.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_rscov_v1/torch_rowcuts_v15.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_sq_v1/n6y_cutsim_v2.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_sq_v1/n6y_cutsim_v2.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_sq_v1/pbase_cutsim_v1.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_sq_v1/pbase_cutsim_v1.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_sq_v1/torch_cutsim_s46_v11.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_sq_v1/torch_cutsim_s46_v11.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_sq_v1/torch_push_native_v2.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_sq_v1/torch_push_native_v2.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_sq_v1/torch_rowcuts_v15.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_sq_v1/torch_rowcuts_v15.py)
- [code/astra_hunt_20261013_v1/winograd_review/check_rlf_canonical_v1.py](code/astra_hunt_20261013_v1/winograd_review/check_rlf_canonical_v1.py)
- [code/astra_hunt_20261013_v1/winograd_review/fragment_rlf_canonical_v1.py](code/astra_hunt_20261013_v1/winograd_review/fragment_rlf_canonical_v1.py)
- [code/astra_hunt_20261013_v1/winograd_review/native_nod96_v1/candidate.py](code/astra_hunt_20261013_v1/winograd_review/native_nod96_v1/candidate.py)
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
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k2tw_x/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k2tw_x/candidate.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/make_k3_v7.py](code/opus_reexam_20261001_v1/scout_card/make_k3_v7.py)
- [code/opus_reexam_20261001_v1/scout_card/make_rb_v1.py](code/opus_reexam_20261001_v1/scout_card/make_rb_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1932](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1932)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1933](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1933)

<details>
<summary>All result files and records (4)</summary>

- [REPORT.md](../../../records/research/2026-10-13-reassociated-push-read-algebra-adopted-0-195-flops-rounding/REPORT.md) · result
- [WINDDOWN_HANDOFF_20261013T043104Z.txt](WINDDOWN_HANDOFF_20261013T043104Z.txt) · result
- [OPUS_HANDOFF_20261013.txt](../../../records/research/2026-10-13-reassociated-push-read-algebra-adopted-0-195-flops-rounding/OPUS_HANDOFF_20261013.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-radial-covariance-star-moment-related-variants-null-against-their) · [Copy provenance](../../../PROVENANCE.json)

