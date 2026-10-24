# Distinct third-cumulant dressing is real but closes only 0.05 and 0.01 plane-error points

2026-09-24T21:36Z · experiment · rejected · measured

A three-variable control established the missing transport term. A faithful dense full-network replica showed that the term barely aligned with the actual late-plane error; removing frame truncation also had negligible effect.

Recorded measurement:

    - 2026-09-24 14:40 lead: PHYSICIST CARRY CERTIFICATE (3-variable MC, only the distinct k3(1,2,3) is skewed, correlations .3-.5): true transport factor .2046; the card's linear gate .1789 (12.6% LOW); the old tensor instrument v4 .1772 (it skips bare-vertex bond patterns, so it never contained the dressing term; memory corrected); linear + first-order dressing d_i d_j C_ij g_k .2120; exact E[f'f'f'] .2105. So the dressing is real and first order captures nearly all of it. Dense A/B sizing on MPS (float32, watchdog on swap/memory) vs the 8M truth kink planes at L10/L14 is running. If B closes the card's 7.3/8.4% plane error: price cheap carriers (per-layer scalar gate inflation on pushed columns; per-unit g_i(1+beta_i) with beta = rho~ w, O(n^2)/layer; star-read only).
    - 2026-09-24 14:36 [physicist] CARRY TERM SIZED, HYPOTHESIS CLOSED (MPS, r35, dense 1024^3 open-loop replica of the card, validated vs the card's own star/sp to 2e-5 before L7). Kink-block plane error vs 8M truth, noise-corrected, L10/L14: card 7.29/8.39%, dense card bookkeeping without truncation (A) 7.24/8.36%, + distinct-index dressing d_i d_j C_ij g_k (B) 7.19/8.35%. B moves the plane ~1.4% of truth norm but closes only ~0.05/0.01 points. Old-tier truncation is not the source either (A ~ card). Certificate: the true transport is E[f'f'f'] (linear gate 12.6% low at rho .3-.5), and the old v4 instrument's F3 never had the dressing. Details: fable_physicist_v1/leadergap/RESULT.md section J.
    - 2026-09-24 15:00 lead: CARRY HYPOTHESIS CLOSED (physicist §J, dense MPS, r35, kink-block plane error vs 8M truth, noise-corrected): card L10 7.29% / L14 8.39%; A (the card's bookkeeping, dense, no truncation) 7.24 / 8.36; B (A + d_i d_j C_ij k3 g_k on distinct entries) 7.19 / 8.35. The term is real but has cosine ~.1 to the error. Old-tier truncation is NOT the source (A is within .05 pt). NEW INSTRUMENT: dense_carry.py C0 is a faithful 40-s MPS screen for any third-order carrier idea. Next: localize among (S) the star's dropped pushed distinct content, (T) beyond-tree births (triangle), (P) the pair-map bivariate K22/K31 target.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_physicist_v1/refit6b/carry/dense_carry.py](code/fable_physicist_v1/refit6b/carry/dense_carry.py)
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
<summary>Additional programs and dependencies (16)</summary>

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
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L409](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L409)
- [TEAM_LOG_2026-09-22_to_09-26.md#L414](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L414)
- [TEAM_LOG_2026-09-22_to_09-26.md#L416](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L416)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../../late-layer-exactness/layer-15-third-cumulant-response-cut-z-3-5/RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-distinct-third-cumulant-dressing-real-but-closes-only-0) · [Copy provenance](../../../PROVENANCE.json)

