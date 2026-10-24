# Isotropic (3,1)-plane coefficient scan is flat near 0.65-0.69; no new dial change

2026-09-25T09:17Z · experiment · rejected · measured

Changing the plane coefficient alone did not reproduce the small gain seen during the joint refit. Its drift was a coordinated change with the learned parameters.

Recorded measurement:

    - 2026-09-25 02:17 [physicist] c0 LINE SEARCH LAUNCHED (lead GO): no-refit VALID16, CPU float64, S24-structure torch card refit6b/s6bcard_loop5j.py (= loop4j with S24's star-Nystrom 'nysp' via the porter's install_star; lvfuse2 is bit-identical), loopR3 params, S24 dials with only c0 changed: S24 (.6944) / .65 / .60 / .55 / .50 (refit6b/c0scan/). Rule (lead): the best point must beat S24 by >= 0.3% to become a c0-only dial literal for S25; a smooth curve with a clear minimum wanted; if monotone to .50, report without extrapolating. ETA ~02:32.
    - 2026-09-25 02:26 [physicist] c0 LINE SEARCH: NO S25 CHANGE (no refit, VALID16, CPU float64, S24-structure card s6bcard_loop5j, loopR3 params, S24 dials with only c0 changed; refit6b/c0scan/c0scan_*/eval_n16_cpu.json). vs S24 (c0 .6944, 1.64837e-8): c0 .65 -0.011% (6/16), .60 +0.086% (5/16), .55 +0.423% (5/16), .50 +0.967% (4/16). Smooth, no cliff, flat minimum at .65-.69: S24's c0 is already at the no-refit optimum; the loopR4 c0 drift is a joint co-move with the learned parts, worth -0.18% in total. No point beats S24 by >= 0.3%. Side check for the lead's structure question: the S24 star-Nystrom mirror (porter's nysp, loop5j) vs loop4j on the same literals = +0.052% pooled, so the loopR4 run on loop4j is representative.
    - 2026-09-25 02:27 lead: c0 LINE SEARCH CLOSED (physicist, no refit, CPU VALID16 on the S24-structure card): c0 .65 -0.011%, .60 +0.086%, .55 +0.42%, .50 +0.97% vs S24 (.6944). The minimum is flat across .65-.69. loopR4's drift was a joint co-move worth -0.18%. The loopR4 verdict holds on the S24 structure (+0.05%).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_physicist_v1/refit6b/s6bcard_loop5j.py](code/fable_physicist_v1/refit6b/s6bcard_loop5j.py)
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

<details>
<summary>Additional programs and dependencies (23)</summary>

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
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_covhunter_v1/s6bcard_joint2t.py](code/fable_covhunter_v1/s6bcard_joint2t.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L944](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L944)
- [TEAM_LOG_2026-09-22_to_09-26.md#L947](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L947)
- [TEAM_LOG_2026-09-22_to_09-26.md#L948](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L948)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-isotropic-3-1-plane-coefficient-scan-flat-near-0) · [Copy provenance](../../../PROVENANCE.json)

