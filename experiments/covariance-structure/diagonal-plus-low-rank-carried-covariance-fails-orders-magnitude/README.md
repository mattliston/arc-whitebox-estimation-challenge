# Diagonal-plus-low-rank carried covariance fails by orders of magnitude even at rank 512

2026-09-25T02:07Z · experiment · rejected · measured

Eigen-truncation with exact variance restoration destroyed final accuracy, including in the late-layer-only arm. The diagonal term also prevented a compelling arithmetic saving.

Recorded measurement:

    - 2026-09-24 19:07 [engineer] COVARIANCE DIAG+LOW-RANK KILLED (E-63, cov_lowrank_s18.py; S18-physics torch card = j05 + star hook, no refit, 8 VALID nets): covariance leaving layers 0-14 -> rank-r eigen-truncation with the diagonal reset to the exact variances. Pooled final MSE: r512 +10,284% (~100x), r256 +46,670%, r128 +183,656%; 0/8 better at every rank (base reproduces the certified hook value exactly). The bill case was negative anyway (the diagonal part still needs the full weighted Gram W^T D W). Covariance must stay exact.
    - 2026-09-24 19:42 lead: COVARIANCE diag+low-rank KILLED (engineer, S18-physics torch card, 8 VALID nets, no refit): r512 about 100x, r256 about 470x, r128 about 1,800x worse final MSE, 0/8. The covariance is precision-critical at every rank (unlike the star plane), and the bill side was negative anyway. The torch star hook is certified against the native fragment at function level (all 14 planes of net 37: max rel 1.6e-6). Refit-step profile running.
    - 2026-09-24 20:03 [engineer] (1) PREBIND FALLBACK FORCED ONCE (lead's S17 condition; prebind_fallback_test.py, fired S17 file in process): setup bound 358; the arena was then replaced by an object whose indexing raises; the first predict logged 'arena lost after setup: lazy binding', completed, and its predictions are BYTE-IDENTICAL to S17 without prebind (bill = the ordinary cold bill, 219 entries rebound lazily). PASS. (2) COVARIANCE LATE ARM (layers 8-14 only, 8 VALID nets): r512 +393% (~5x), r256 +3,282%; 0/8 better at both. Kill confirmed at every layer range. (3) Starting the computed-but-unused audit of S18 with a gradient-liveness probe on the torch card.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/cov_lowrank_s18.py](code/fable_engineer_v1/cov_lowrank_s18.py)
- [code/fable_engineer_v1/prebind_fallback_test.py](code/fable_engineer_v1/prebind_fallback_test.py)
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

<details>
<summary>Additional programs and dependencies (20)</summary>

- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_engineer_v1/lrstar_torch_hook.py](code/fable_engineer_v1/lrstar_torch_hook.py)
- [code/fable_physicist_v1/refit6b/carry/variant_bd.py](code/fable_physicist_v1/refit6b/carry/variant_bd.py)
- [code/fable_physicist_v1/refit6b/s6bcard_j05.py](code/fable_physicist_v1/refit6b/s6bcard_j05.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L573](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L573)
- [TEAM_LOG_2026-09-22_to_09-26.md#L574](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L574)
- [TEAM_LOG_2026-09-22_to_09-26.md#L597](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L597)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-diagonal-plus-low-rank-carried-covariance-fails-orders-magnitude) · [Copy provenance](../../../PROVENANCE.json)

