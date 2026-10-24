# Layer-7 rank-cut distillation failed both a fixed gradient point and four-direction least squares

2026-10-18T15:43:20Z · experiment · rejected · measured

A rank-320 student with cloned late encoders was distilled against the rank-384 teacher using four fixed training networks. The first point increased teacher loss; a bounded four-direction Gauss-Newton fit explained too little discrepancy and closed the entire branch without a check set.

Recorded measurement:

    First point: teacher loss +180.33%, corrected error +8.088%; Gauss–Newton 20 forwards, float32 least squares explains 1.695678% <25% gate; condition 10.37.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

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

<details>
<summary>Additional programs and dependencies (50)</summary>

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
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v8.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v8.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/offline_train8_v1/guard_offline9gb_free50_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/offline_train8_v1/guard_offline9gb_free50_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/grade_point_train4_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/grade_point_train4_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/inspect_student_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/inspect_student_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/run_fixed4_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/run_fixed4_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/run_forward_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/run_forward_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/run_one_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/run_one_v3.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/run_point_train4_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_feasibility_v1/run_point_train4_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_gn_v1/build_basis_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_gn_v1/build_basis_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_gn_v1/run_probes_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_gn_v1/run_probes_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_gn_v1/solve_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_distill_gn_v1/solve_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_encoder_distill_v1/candidate_95c_L7rank320_encoder7_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_encoder_distill_v1/candidate_95c_L7rank320_encoder7_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_encoder_distill_v1/late_encoder_scope_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_encoder_distill_v1/late_encoder_scope_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/current_torch_jsk_port_v1/torch_w3_current95c_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/current_torch_jsk_port_v1/torch_w3_current95c_v2.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py)
- [code/opus_reexam_20261001_v1/trainer/otr_gop2_v1.py](code/opus_reexam_20261001_v1/trainer/otr_gop2_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m_v9.py](code/opus_reexam_20261001_v1/trainer/otr_m_v9.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl_v2.py](code/opus_reexam_20261001_v1/trainer/otr_pl_v2.py)
- [code/opus_reexam_20261001_v1/trainer/worker_generated_v18.py](code/opus_reexam_20261001_v1/trainer/worker_generated_v18.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5706](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5706)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5708](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5708)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5710](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5710)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [CLOSURE_v1.json](CLOSURE_v1.json) · result
- [MANIFEST_v1.json](MANIFEST_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-layer-7-rank-cut-distillation-failed-both-fixed-gradient) · [Copy provenance](../../../PROVENANCE.json)

