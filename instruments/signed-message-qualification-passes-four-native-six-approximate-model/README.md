# Signed-message qualification passes four native and six approximate-model calls with exact repeated gradient bytes

2026-09-21 · tool · informational · measured

The signed-message extension passed native control and cold/warm identity checks plus fixed approximate-model gradient and zero-learning-rate optimizer checks. Own candidate bills were measured, but no native derivative, training result or predictive gain was certified.

Recorded measurement:

    4fullnativecalls;6approximate-model records;repeat_gradient_bytes_exact=true;allparameters unchanged;totalwall252.482805875s;approximate peak5786730496B.

## Programs

- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/allocator.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/allocator.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/common.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/common.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/gnn_step.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/gnn_step.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/guard.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/guard.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/model_branch.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/model_branch.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/model_parent.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/model_parent.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/native.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/native.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/prepare.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/prepare.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/qualify_native.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/qualify_native.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/qualify_torch.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/qualify_torch.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/records.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/records.py)

<details>
<summary>Additional programs and dependencies (23)</summary>

- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/run.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/run.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/unit_native.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/unit_native.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/unit_torch.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/unit_torch.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/guard.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/guard.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/guard.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/common.py](code/codex_resume_20260919_v1/current_p_release_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/native.py](code/codex_resume_20260919_v1/current_p_release_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/records.py](code/codex_resume_20260919_v1/current_p_release_v1/records.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [result.json#L1](result.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [result.json](result.json) · record
- [native_qualification.json](native_qualification.json) · record
- [torch_qualification.json](torch_qualification.json) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-21-signed-message-qualification-passes-four-native-six-approximate-model) · [Copy provenance](../../PROVENANCE.json)

