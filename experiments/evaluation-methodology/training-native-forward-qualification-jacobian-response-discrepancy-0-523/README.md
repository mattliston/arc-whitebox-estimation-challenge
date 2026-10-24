# Training/native forward qualification: Jacobian response discrepancy 0.523 fails the gate

2026-09-29 · experiment · rejected · measured

The zero native control matched submission#332838 exactly. After a scalar-wrapper technical repair, the training implementation still failed its response qualification, so no second network or certified fit followed.

Recorded measurement:

    Jacobian-vector relative discrepancy.523;native half-step stability.508/.576 versus.5 threshold;native35 zero prediction byte-exact.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/assess.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/assess.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/assess_runtime_v2.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/assess_runtime_v2.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/direction_common.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/direction_common.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/native_parameters.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/native_parameters.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/native_probe.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/native_probe.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/torch_probe.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/torch_probe.py)
- [code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/torch_runtime_v2.py](code/codex_resume_20260926_v2/s33_joint_forward_qualification_v1/torch_runtime_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)

<details>
<summary>Additional programs and dependencies (23)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py)
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

- [handoff_root_archive_20260930T024527Z.md#L106](../../arithmetic-kernels/depth-three-covariance-diagonal-products-adjusted-gain-0-355945/handoff_root_archive_20260930T024527Z.md#L106)

<details>
<summary>All result files and records (2)</summary>

- [handoff_root_archive_20260930T024527Z.md](../../arithmetic-kernels/depth-three-covariance-diagonal-products-adjusted-gain-0-355945/handoff_root_archive_20260930T024527Z.md) · record
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-29-training-native-forward-qualification-jacobian-response-discrepancy-0-523) · [Copy provenance](../../../PROVENANCE.json)

