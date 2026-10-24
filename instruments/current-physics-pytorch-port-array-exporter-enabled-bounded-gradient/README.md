# Current-physics PyTorch port and array exporter enabled bounded gradient tests with explicit parity checks

2026-10-18T13:56:09Z · tool · informational · measured

A private PyTorch port incorporated current join sketches, exact cubic messages and read-tail behavior, with forward and gradient checks. Mutable metadata replay was repaired, and the array-only exporter reproduced the frozen estimator byte for byte with zero updates; offline memory limits remained separate from native release limits.

Recorded measurement:

    Two forwards: ~90% squared discrepancy reduction; selected 14-array backward: 57.19 s, peak 8.178 GB; zero-update 95c byte round-trip PASS.

## Programs

- [code/opus_reexam_20261001_v1/lead/runs/astra_gop2_carry_absorb_held_v1/trainer/worker_generated_v18.py](code/opus_reexam_20261001_v1/lead/runs/astra_gop2_carry_absorb_held_v1/trainer/worker_generated_v18.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1/candidate_v61j.py](code/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1/candidate_v61j.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/candidate_95c_fusion_strip5_sym_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/candidate_95c_fusion_strip5_sym_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/n6y_rtt_private_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/n6y_rtt_private_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_inner_checkpoint_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_inner_checkpoint_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rtt_current95c_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rtt_current95c_v3.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rtt_current95c_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rtt_current95c_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v8.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v8.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_batch_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_batch_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_replay_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_replay_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/diagnose_train8_loo_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/diagnose_train8_loo_v1.py)

<details>
<summary>Additional programs and dependencies (97)</summary>

- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v3.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rtt_diagnostic_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rtt_diagnostic_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_fixed_check8_forward_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_fixed_check8_forward_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_port_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_port_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rowcuts_rtt_private_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rowcuts_rtt_private_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_core_moment_chunks_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_core_moment_chunks_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_core_moment_checkpoint_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_core_moment_checkpoint_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v4.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v4.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v7.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v7.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/core_moment_checkpoint_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/core_moment_checkpoint_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_plain_capture_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_plain_capture_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rowcuts_rtt_private_v5.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rowcuts_rtt_private_v5.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_push_current95c_rtt_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_push_current95c_rtt_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v5.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v5.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v6.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v6.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_forward_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_forward_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rowcuts_rtt_private_v4.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rowcuts_rtt_private_v4.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v5.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v5.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v6.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v6.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_push_current95c_rtt_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_push_current95c_rtt_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_forward_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_forward_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/diagnose_mean_gradient_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/diagnose_mean_gradient_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_gradient_memory_diagnostic_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_gradient_memory_diagnostic_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/build_private_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/build_private_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_fixed_train_forward_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_fixed_train_forward_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rowcuts_rtt_private_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rowcuts_rtt_private_v3.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v3.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_fixed_check8_forward_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_fixed_check8_forward_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_diagnostic_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_diagnostic_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rowcuts_rtt_private_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rowcuts_rtt_private_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_core_moment_chunks_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/tiny_core_moment_chunks_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v4.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_v4.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v7.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v7.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/core_moment_checkpoint_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/core_moment_checkpoint_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rtt_current95c_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/torch_rtt_current95c_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v8.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v8.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_batch_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/run_current_gradient_batch_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/n6y_rtt_private_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/n6y_rtt_private_v1.py)
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
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_gop2_carry_absorb_held_v1/trainer/otr_gop2_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_gop2_carry_absorb_held_v1/trainer/otr_gop2_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_gop2_carry_absorb_held_v1/trainer/otr_m_v9.py](code/opus_reexam_20261001_v1/lead/runs/astra_gop2_carry_absorb_held_v1/trainer/otr_m_v9.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/current_torch_jsk_port_v1/torch_push_current95c_jsk_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/current_torch_jsk_port_v1/torch_push_current95c_jsk_v1.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L5688](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5688)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5690](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5690)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5692](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5692)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [CURRENT_TORCH_PORT_AUDIT_v1.json](CURRENT_TORCH_PORT_AUDIT_v1.json) · record
- [SEARCH_STATE_v7.json](SEARCH_STATE_v7.json) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-18-current-physics-pytorch-port-array-exporter-enabled-bounded-gradient) · [Copy provenance](../../PROVENANCE.json)

