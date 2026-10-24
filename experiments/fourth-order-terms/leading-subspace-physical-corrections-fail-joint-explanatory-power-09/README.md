# Leading-subspace physical corrections fail: joint explanatory power <= .09

2026-09-23T05:58Z · experiment · rejected · measured

Gain-skew and plane-derived physical rules aligned weakly with the needed covariance correction; using truth third-order planes weakened the apparent alignment.

Recorded measurement:

    2026-09-22 22:58 physicist: TOP-K physical rule: NONE (fable_physicist_v1/topk/RESULT_TOPK.md). r35 rho with R (k=30, L9-14): P1 .20-.26, gain-skew .15-.18, tree31 .11-.16, gain-skew K22 negative; joint R2 <= .09; TRUTH-K21 versions of P1/gsk are WORSE (alignment = card's own K21 error, not physics). Closed loop top-k block: P1(.9) eps .25/.5 -0.16/-0.18% (r39 +1.3%), gsk +0.5/+1.3%. k-direction reads of carried objects add no information (card already forms full K21 from them); lever = non-isotropic 4th order, trees capture ~12%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

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
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)

<details>
<summary>Additional programs and dependencies (11)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_physicist_v1/model_bank_audit.py](code/fable_physicist_v1/model_bank_audit.py)
- [code/fable_physicist_v1/topk/topk_closedloop.py](code/fable_physicist_v1/topk/topk_closedloop.py)
- [code/fable_physicist_v1/topk/topk_screen.py](code/fable_physicist_v1/topk/topk_screen.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L78](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L78)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT_TOPK.md](RESULT_TOPK.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-leading-subspace-physical-corrections-fail-joint-explanatory-power-09) · [Copy provenance](../../../PROVENANCE.json)

