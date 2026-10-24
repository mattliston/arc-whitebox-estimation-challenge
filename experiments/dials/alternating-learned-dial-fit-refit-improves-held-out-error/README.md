# Alternating learned-on dial fit and refit improves held-out error 0.89% in the first round

2026-09-25T03:31Z · experiment · adopted · measured

Refitting physics coefficients on the updated low-rank-star trajectory improved validation error. The subsequent learned-parameter refit also passed its own improvement gate.

Recorded measurement:

    2026-09-24 20:31 covhunter: LOOP ROUND 1 DIALS (RESULT sec 28): learned-ON re-fit on the S18-HOOKED card, j05 ep2 params: LOO -3.20%, real -4.12%; VALID16 hooked 1.70220e-8 -> 1.66645e-8 = -2.10% (14/16). Dials: j1 .559, j2 +.358, j3 +.152, b 1.0509, carried 1.00469/layer, newborn .99312, c0 .709, c1 .503, k4 .989/.963/.960 (j3, c1 flat across fits). Module s6bcard_jfitR1.py (joint2 semantics, STAR lr32) handed to the physicist. VALID40 (ep2+r1) vs S19 (ep0+S19 dials) running (~21:00).
    2026-09-24 20:39 covhunter: LOOP ROUND 1 PASSES (RESULT sec 28): VALID40 (ep2 params + round-1 dials) vs (ep0 + S19 dials), both S18-hooked shared: -0.89% [CI -1.13,-0.65], 35/40 (VALID16 -0.84% 13/16, other24 -0.92% 22/24); bar >=0.5% per round. Rows for the porter's S21 parity: fable_covhunter_v1/r1_ep2dials_vs_S19.json (fit_s18 = ep2+r1, base_s18 = ep0+S19). Physicist's refit round with s6bcard_jfitR1 next.
    - 2026-09-24 21:47 lead: LOOP ROUND 1 REFIT PASSES (physicist s6b_loopR1): VALID16 1.650543e-8 vs its own start 1.666309e-8 = -0.95% (bar -0.5%, <= 1.657977e-8). params sha f18a30b9... Covhunter's round-2 dial refit auto-started; S22 waits on dials_r2_loopR1.json. Per-round gains so far: dials R1 -0.89% (VALID40), refit R1 -0.95% (VALID16).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_covhunter_v1/s6bcard_jfitR1.py](code/fable_covhunter_v1/s6bcard_jfitR1.py)
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
<summary>Additional programs and dependencies (18)</summary>

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
- [code/fable_engineer_v1/lrstar_torch_hook.py](code/fable_engineer_v1/lrstar_torch_hook.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L609](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L609)
- [TEAM_LOG_2026-09-22_to_09-26.md#L617](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L617)
- [TEAM_LOG_2026-09-22_to_09-26.md#L668](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L668)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [r1_ep2dials_vs_S19.json](r1_ep2dials_vs_S19.json) · result

</details>

## Related work

- shipped in: [Submission #332231: epoch-2 learned parameters with first-round physics scales](../../../submissions/332231/README.md)
- shipped in: [Submission #332241: one-epoch learned refit at the fitted physics scales](../../../submissions/332241/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-alternating-learned-dial-fit-refit-improves-held-out-error) · [Copy provenance](../../../PROVENANCE.json)

