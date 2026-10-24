# Born-unit plane-to-distinct feed at half strength improves learned-off raw error 6.3%

2026-09-24T22:42Z · experiment · adopted · measured

The unorthogonalized born-unit-side plane feed aligns with the distinct-content error better than the existing learned orthogonalized term. It improves physical moments but initially worsens the learned-on estimator.

Recorded measurement:

    - 2026-09-24 15:42 lead: PHYSICIST ALIGNMENT SCREEN FOUND THE TERM. j1 = the born-unit-side plane-to-distinct feed -(g_i/2) eta_a K21[a,i]. At physical strength its cosine to the distinct error is +.40-.49 (L6-14), and one layer cuts the distinct error 7.43->6.76% (L10). The card's learned j1 coefficient is only .13-.33, on an ORTHOGONALIZED direction; the parallel component matters. No refit, r35: j1 X=1 (unorthogonalized) plane 5.23/5.92%, k3 3.71/3.92% (-2.0/-2.5 pt, passes the 2-pt bar), but MSE +40% learned-ON (cancellation) and +3.5% learned-OFF; X=.5 MSE +4.6% ON / -5.6% OFF. VALID16 learned-OFF X=.5: -6.3%, 15/16. DECISION: the refit slot (after wide2x) goes to j1 X=.5 (or combo with b x .9 if the combo beats j1 alone by >= .5% learned-off). b x .9 alone moves to the second slot after the T640 trunk (~17:50). Bar VALID16 <= 1.7374e-8.
    - 2026-09-24 15:44 [physicist] ALIGNMENT SCREENS: the aligned distinct-content term is the born-unit-side plane feed j1 (learned at c1 0.13-0.33 on the orthogonalized direction). At physical form: cos +0.40..+0.49 to the distinct error (L6-14); one layer takes the distinct error 7.43 -> 6.76% (L10). No refit, r35: j1 physical X=1 plane 5.23/5.92% (-2.0/-2.5 pt), k3 3.71/3.92%, but MSE +40% learned on. Learned-OFF VALID16: j1 X=0.5 0.9367 (-6.3%, 15/16); b x0.9 0.9729; j1 0.5 + b 0.9 0.9650 (not additive). Recommended j1 X=0.5 for the refit slot (module refit6b/s6bcard_j05.py validated); awaiting the lead's word, and the b x0.9 launcher fires as approved otherwise. RESULT.md section L.
    - 2026-09-24 15:45 [physicist] PRE-REGISTERED (lead: swap to j05; b x0.9 parked, revisit only on top of a j1-refit card): j1 X=0.5 REFIT. B2 recipe (groups full+P, lr 1e-4 / trunk 5e-5, warmup 3/5, clip 1, batch 4), 1 epoch TRAIN160, warm start from B2 ep1, card = refit6b/s6bcard_j05.py (s6bcard with the born-unit-side leg j1 in physical form, unorthogonalized, at 0.5; replaces the learned c1*orth term). BAR: VALID16 <= 1.7374e-8 (-0.5% vs B2 ep1 1.74614e-8); if met, VALID40. The no-refit learned-on starting point is the run's own valid_before (r35: +4.6%). The b09 launcher was killed before it fired. Starts when covhunter's wide2x (pid 76221) exits, 5 threads, nice 10.
    - 2026-09-24 16:10 [physicist] j05 refit STARTED 15:50 (s6b_B2ep1_j05_1ep). No-refit learned-ON starting point (the run's valid_before, j05 card at B2 ep1 literals, VALID16): 1.83290e-8 = +4.97% vs B2 ep1 1.74614e-8 (r35 was +4.6%). Bar: <= 1.7374e-8.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_physicist_v1/refit6b/s6bcard_j05.py](code/fable_physicist_v1/refit6b/s6bcard_j05.py)
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
<summary>Additional programs and dependencies (17)</summary>

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
- [code/fable_physicist_v1/refit6b/carry/variant_bd.py](code/fable_physicist_v1/refit6b/carry/variant_bd.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L449](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L449)
- [TEAM_LOG_2026-09-22_to_09-26.md#L450](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L450)
- [TEAM_LOG_2026-09-22_to_09-26.md#L451](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L451)
- [TEAM_LOG_2026-09-22_to_09-26.md#L459](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L459)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332224: half-strength unit birth feed, refit, lean views, and setup prebinding](../../../submissions/332224/README.md)

[Research graph](../../../graph/index.html#node=2026-09-24-born-unit-plane-distinct-feed-half-strength-improves-learned) · [Copy provenance](../../../PROVENANCE.json)

