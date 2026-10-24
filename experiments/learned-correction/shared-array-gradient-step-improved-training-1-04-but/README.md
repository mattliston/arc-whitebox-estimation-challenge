# Shared-array gradient step improved training by 1.04% but only 0.049587% on untouched checks

2026-10-18T14:37:21Z · experiment · rejected · measured

A single gradient direction from eight training networks was scaled to a one-percent target without a line search. Its training improvement did not generalize to eight untouched check networks, and leave-one-out training diagnostics identified selection noise.

Recorded measurement:

    TRAIN geometric -1.040273%, pooled -1.051297%; CHECK geometric -.049587%, pooled -.062889%, SE .230837 percentage points, 3/8 wins; leave-one-out predicts +.1279%; conditional 1.446139099e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/fixed_check8_v1/run_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/fixed_check8_v1/run_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/fixed_check8_v1/run_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/fixed_check8_v1/run_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/fixed_check8_v1/grade_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/fixed_check8_v1/grade_v1.py)
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
<summary>Additional programs and dependencies (44)</summary>

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
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v8.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v8.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_fixed_check8_forward_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_fixed_check8_forward_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_fixed_check8_forward_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_fixed_check8_forward_v2.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L5694](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5694)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5696](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5696)

<details>
<summary>All result files and records (22)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [CLOSURE_v1.json](CLOSURE_v1.json) · result
- [RUN_PLAN_v1.json](RUN_PLAN_v1.json) · result
- [GUARD_7103476_point_v1.json](GUARD_7103476_point_v1.json) · result
- [GUARD_7103473_control_v1.json](GUARD_7103473_control_v1.json) · result
- [GUARD_7103472_point_v1.json](GUARD_7103472_point_v1.json) · result
- [GUARD_7103475_point_v1.json](GUARD_7103475_point_v1.json) · result
- [GUARD_7103474_control_v1.json](GUARD_7103474_control_v1.json) · result
- [GUARD_7103475_control_v1.json](GUARD_7103475_control_v1.json) · result
- [PAIRS_v1.json](PAIRS_v1.json) · result
- [GUARD_7103478_point_v1.json](GUARD_7103478_point_v1.json) · result
- [GUARD_7103472_control_v1.json](GUARD_7103472_control_v1.json) · result
- [GUARD_7103477_control_v1.json](GUARD_7103477_control_v1.json) · result
- [GUARD_7103479_point_v1.json](GUARD_7103479_point_v1.json) · result
- [GUARD_7103478_control_v1.json](GUARD_7103478_control_v1.json) · result
- [GUARD_7103474_point_v1.json](GUARD_7103474_point_v1.json) · result
- [GRADE_v1.json](GRADE_v1.json) · result
- [GUARD_7103476_control_v1.json](GUARD_7103476_control_v1.json) · result
- [GUARD_7103473_point_v1.json](GUARD_7103473_point_v1.json) · result
- [GUARD_7103479_control_v1.json](GUARD_7103479_control_v1.json) · result
- [GUARD_7103477_point_v1.json](GUARD_7103477_point_v1.json) · result
- [TRAIN_SCREEN_v1.json](TRAIN_SCREEN_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-shared-array-gradient-step-improved-training-1-04-but) · [Copy provenance](../../../PROVENANCE.json)

