# Nonlinear covariance response-input fit: native held-out training subset worsens 0.143173%

2026-09-27 · experiment · rejected · measured

Two zero-initialized12-by-16 matrices were fitted for12 updates with prior parameters frozen and native residual offsets. Fit gain partly transferred, but the fixed CHECK8 generalization gate failed against the local-feature control.

Recorded measurement:

    FIT12/CHECK8 within TRAIN160;message raw+0.072316%,adjusted+0.143173%,2/8 wins,95% CI[-0.025314551582122924,0.2876543309141555];control adjusted-0.065038%,4/8 wins;message/control+0.208347%;FIT38/40 raw-0.90702%/-0.57648%,response discrepancies55.9%/72.3%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/assess.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/assess.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/assess_before_json_repair_v1.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/assess_before_json_repair_v1.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/assess_before_matched_parity_v1.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/assess_before_matched_parity_v1.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/build_native.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/build_native.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/capture.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/capture.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/diagnose_fit_transfer.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/diagnose_fit_transfer.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/finish_pilot.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/finish_pilot.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/fit.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/fit.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/fit_transfer.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/fit_transfer.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/match_baseline.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/match_baseline.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/model.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/model.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/native_zero.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/native_zero.py)

<details>
<summary>Additional programs and dependencies (34)</summary>

- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/qualify.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/qualify.py)
- [code/codex_resume_20260926_v2/same_layer_edge_messages_v1/record_checkpoint.py](code/codex_resume_20260926_v2/same_layer_edge_messages_v1/record_checkpoint.py)
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
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py)
- [code/codex_resume_20260926_v2/candidate_same_layer_edge_control_terminal_v1.py](code/codex_resume_20260926_v2/candidate_same_layer_edge_control_terminal_v1.py)
- [code/codex_resume_20260926_v2/candidate_same_layer_edge_message_terminal_v1.py](code/codex_resume_20260926_v2/candidate_same_layer_edge_message_terminal_v1.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)
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

</details>

## Evidence

- [RESULTS.md#L1142](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1142)

<details>
<summary>All result files and records (3)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [assessment_matched.json](assessment_matched.json) · result
- [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-nonlinear-covariance-response-input-fit-native-held-out-training) · [Copy provenance](../../../PROVENANCE.json)

