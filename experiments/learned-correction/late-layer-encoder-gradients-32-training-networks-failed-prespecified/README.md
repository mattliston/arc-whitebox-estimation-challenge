# Late-layer encoder gradients on 32 training networks failed the prespecified global signal gate

2026-10-18T14:49:23Z · experiment · rejected · measured

Six encoder arrays were cloned only for layers 12–15 while original arrays and cubic messages stayed fixed. Thirty-two training gradients completed within the offline envelope, but their global signal failed the two-jackknife-standard-error gate, so no parameter point or check run was made.

Recorded measurement:

    Signal -4.65777e-19 +/-4.83215e-19 SE; required >9.66430e-19; peak 7.359 GB / free 58.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/torch_late_encoder_v1/train32_v1/run_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/torch_late_encoder_v1/train32_v1/run_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/torch_late_encoder_v1/late_encoder_scope_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/torch_late_encoder_v1/late_encoder_scope_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/torch_late_encoder_v1/run_late_encoder_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/torch_late_encoder_v1/run_late_encoder_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/torch_late_encoder_v1/run_late_encoder_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/torch_late_encoder_v1/run_late_encoder_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/torch_late_encoder_v1/unit_scope_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/torch_late_encoder_v1/unit_scope_v1.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)

<details>
<summary>Additional programs and dependencies (45)</summary>

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
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v8.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/configure_private_v8.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/offline_train8_v1/guard_offline9gb_free50_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/offline_train8_v1/guard_offline9gb_free50_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/late_encoder_design_v1/candidate_95c_late_encoder12_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/late_encoder_design_v1/candidate_95c_late_encoder12_HELD_v1.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L5696](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5696)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5698](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5698)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5700](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5700)

<details>
<summary>All result files and records (68)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUMMARY_v1.json](SUMMARY_v1.json) · result
- [GUARD_7100271_v1.json](GUARD_7100271_v1.json) · result
- [COMMAND_7100291_v1.json](COMMAND_7100291_v1.json) · result
- [COMMAND_7100281_v1.json](COMMAND_7100281_v1.json) · result
- [COMMAND_7100275_v1.json](COMMAND_7100275_v1.json) · result
- [GUARD_7100285_v1.json](GUARD_7100285_v1.json) · result
- [GUARD_7100295_v1.json](GUARD_7100295_v1.json) · result
- [COMMAND_7100288_v1.json](COMMAND_7100288_v1.json) · result
- [COMMAND_7100298_v1.json](COMMAND_7100298_v1.json) · result
- [GUARD_7100278_v1.json](GUARD_7100278_v1.json) · result
- [COMMAND_7100299_v1.json](COMMAND_7100299_v1.json) · result
- [COMMAND_7100289_v1.json](COMMAND_7100289_v1.json) · result
- [GUARD_7100269_v1.json](GUARD_7100269_v1.json) · result
- [GUARD_7100279_v1.json](GUARD_7100279_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [GUARD_7100270_v1.json](GUARD_7100270_v1.json) · result
- [COMMAND_7100280_v1.json](COMMAND_7100280_v1.json) · result
- [COMMAND_7100290_v1.json](COMMAND_7100290_v1.json) · result
- [COMMAND_7100274_v1.json](COMMAND_7100274_v1.json) · result
- [GUARD_7100294_v1.json](GUARD_7100294_v1.json) · result
- [GUARD_7100284_v1.json](GUARD_7100284_v1.json) · result
- [COMMAND_7100292_v1.json](COMMAND_7100292_v1.json) · result
- [COMMAND_7100282_v1.json](COMMAND_7100282_v1.json) · result
- [ROOT_GO_v1.json](ROOT_GO_v1.json) · result
- [GUARD_7100272_v1.json](GUARD_7100272_v1.json) · result
- [GUARD_7100286_v1.json](GUARD_7100286_v1.json) · result
- [GUARD_7100296_v1.json](GUARD_7100296_v1.json) · result
- [COMMAND_7100276_v1.json](COMMAND_7100276_v1.json) · result
- [COMMAND_7100283_v1.json](COMMAND_7100283_v1.json) · result
- [COMMAND_7100293_v1.json](COMMAND_7100293_v1.json) · result
- [GUARD_7100273_v1.json](GUARD_7100273_v1.json) · result
- [GUARD_7100297_v1.json](GUARD_7100297_v1.json) · result
- [GUARD_7100287_v1.json](GUARD_7100287_v1.json) · result
- [COMMAND_7100277_v1.json](COMMAND_7100277_v1.json) · result
- [GRADIENT_MANIFEST_v1.json](GRADIENT_MANIFEST_v1.json) · result
- [GUARD_7100282_v1.json](GUARD_7100282_v1.json) · result
- [GUARD_7100292_v1.json](GUARD_7100292_v1.json) · result
- [COMMAND_7100272_v1.json](COMMAND_7100272_v1.json) · result
- [COMMAND_7100296_v1.json](COMMAND_7100296_v1.json) · result
- [COMMAND_7100286_v1.json](COMMAND_7100286_v1.json) · result
- [GUARD_7100276_v1.json](GUARD_7100276_v1.json) · result
- [GUARD_7100293_v1.json](GUARD_7100293_v1.json) · result
- [GUARD_7100283_v1.json](GUARD_7100283_v1.json) · result
- [COMMAND_7100273_v1.json](COMMAND_7100273_v1.json) · result
- [COMMAND_7100287_v1.json](COMMAND_7100287_v1.json) · result
- [COMMAND_7100297_v1.json](COMMAND_7100297_v1.json) · result
- [GUARD_7100277_v1.json](GUARD_7100277_v1.json) · result
- [KNOWN_BASELINE_PARITY_v1.json](KNOWN_BASELINE_PARITY_v1.json) · result
- [COMMAND_7100271_v1.json](COMMAND_7100271_v1.json) · result
- [GUARD_7100281_v1.json](GUARD_7100281_v1.json) · result
- [GUARD_7100291_v1.json](GUARD_7100291_v1.json) · result
- [GUARD_7100275_v1.json](GUARD_7100275_v1.json) · result
- [COMMAND_7100295_v1.json](COMMAND_7100295_v1.json) · result
- [COMMAND_7100285_v1.json](COMMAND_7100285_v1.json) · result
- [GUARD_7100298_v1.json](GUARD_7100298_v1.json) · result
- [GUARD_7100288_v1.json](GUARD_7100288_v1.json) · result
- [COMMAND_7100278_v1.json](COMMAND_7100278_v1.json) · result
- [GUARD_7100289_v1.json](GUARD_7100289_v1.json) · result
- [GUARD_7100299_v1.json](GUARD_7100299_v1.json) · result
- [COMMAND_7100279_v1.json](COMMAND_7100279_v1.json) · result
- [COMMAND_7100269_v1.json](COMMAND_7100269_v1.json) · result
- [COMMAND_7100270_v1.json](COMMAND_7100270_v1.json) · result
- [GUARD_7100290_v1.json](GUARD_7100290_v1.json) · result
- [GUARD_7100280_v1.json](GUARD_7100280_v1.json) · result
- [GUARD_7100274_v1.json](GUARD_7100274_v1.json) · result
- [COMMAND_7100284_v1.json](COMMAND_7100284_v1.json) · result
- [COMMAND_7100294_v1.json](COMMAND_7100294_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-late-layer-encoder-gradients-32-training-networks-failed-prespecified) · [Copy provenance](../../../PROVENANCE.json)

