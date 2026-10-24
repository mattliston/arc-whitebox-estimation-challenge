# Rank-cut compensation fails: rescaling, Gram targets and reweighting do not restore lost accuracy

2026-09-25T18:22Z · experiment · rejected · measured

The rank penalty lies in discarded low-energy directions aligned with the existing estimator error. Delaying the cut recovered part of the loss but gave back much of the work saving.

Recorded measurement:

    - 2026-09-25 11:25 covhunter: RANK-320 COMPENSATION: FAIL, no cheap stand-in exists (RESULT sec 35; fable_covhunter_v1/r320_comp7.py; torch loopR3 card, 384/320/320; screen TRAIN24 = every 6th TRAIN160 net without panel, measure VALID40). VALID40: schedule raw +2.889% vs all-384 (37/40; reproduces the analyst's +2.88%). (a) scalar on the kept old-tier block, per-join fit s10 1.0022 / s13 0.9981 on TRAIN24 (predicted -0.03 pt): VALID40 +0.078 pt [-0.001, +0.160], recovers 0%; the scale sits at its optimum in a steep bowl (1% costs +1..+2.5 pt). (b) has no target: the tier stores third-order atoms, and its Gram only ranks the frame rows (no covariance or variance consumer); restoring each row's dropped Gram energy = +0.002 pt. (c) value flow, TRAIN24: the cut drops 0.01% of the pairs' common-part energy and 0.4-0.5% of their difference-part energy; frame reweighting toward either part does not help (x0.5 +0.05, x3..30 worse); consumer-specific scales (tier k3 into the mean path, plane read, final k3) best fit predicts -0.04 pt; a 1.03 gain on the learned correction gives +0.06. Timing probe (384 for the join layer's own consumers, cut one layer later) recovers 46% [18, 74]% but gives back 47% of the bill saving (-1.79% vs -3.38%): same exchange rate, so no per-layer timing beats the per-join schedule. Law: the penalty is specific tail directions (low energy, error-aligned); carrying them costs what the cut saves; the rank schedule is the only lever. Shadow beta* not run (nothing passed). For loopR7s: sec 21 refit at all-320 absorbed -0.009% of +3.35%; expect little co-adaptation recovery.
      Covhunter RESULT §35: rank-cut compensation FAILS (scalar at optimum, no Gram target, reweighting null; the timing probe recovers 46% but halves the bill saving). Law: the penalty lives in low-energy tail directions aligned with the error; there is no cheap stand-in.
    - 2026-09-25 11:40 covhunter: L10 CUT + TIMING, torch VALID40 (RESULT sec 35 addendum; the same code and all-384 rows): 384/320/384 (L10 only) raw +1.837% vs all-384 (6/40 better; matches the analyst's additive +1.82%). With the L10 join layer at 384 and the cut from L11: raw +1.096% (7/40 better). That is -0.741 pt [-1.066, -0.432] vs L10 only, 31/40, 40% of the L10 penalty. Native bill (porter): S27r -3.17%, S27s (L13 only) -0.69%, so if the joins add, L10 holds about -2.48%. Estimated adjusted for L10 only is about -0.69% (vs S27r -0.34%, S27s +0.35%). The L11 cut beats L10 only if its native saving stays above 1.77%, i.e. the L10 join layer holds under 29% of that segment's saving (unit-cost model: 50%, but the model missed L13 by 2.7x). Needs native builds: S27t {7:384, 10:320, 13:384} and a per-layer truncation at L11.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_covhunter_v1/r320_comp7.py](code/fable_covhunter_v1/r320_comp7.py)
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
- [code/fable_physicist_v1/refit6b/s6bcard_loop4j.py](code/fable_physicist_v1/refit6b/s6bcard_loop4j.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1106](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1106)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1109](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1109)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1155](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1155)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-rank-cut-compensation-fails-rescaling-gram-targets-reweighting-do) · [Copy provenance](../../../PROVENANCE.json)

