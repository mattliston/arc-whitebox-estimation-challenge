# Consumption-ranked birth pruning: dropping a quarter of late births raises raw error 63%

2026-09-24T17:51Z · experiment · rejected · measured

Ranking births by their next-layer read contribution did not make them safely removable. A star catch recovered only part of the lost signal; all tested pruning arms failed.

Recorded measurement:

    - 2026-09-24 10:51 [physicist] TEST 1 BUDGETED BIRTHS (consumption-ranked: each born pair's read-weighted plane in the NEXT layer from the card's own m, C, W; refit6b/births_prune_eval.py; torch S6b + B2 ep1; no refit; VALID16 vs 1.74614e-8; whole-card bill births_bill.py): keep .75 from L8 raw x1.627 (0/16) bill .975 adj x1.586; keep .5 L8 x5.52 bill .950 adj x5.24; keep .5 L8 + star catch x3.05 adj x2.89; keep .75 from L1 x3.91 bill .874 adj x3.42; keep .5 L1 x20.8 bill .749 adj x15.6. keep .25 not run (keep .5 intolerable). Pruning from L8 cannot cut the bill (-2.5..-7.6%: early cohorts dominate the columns); from L1 the loss is catastrophic. Recommendation: no refit; birth pruning closed.
    - 2026-09-24 10:55 lead: CLOSED lean-card candidates #1 budgeted births (every arm 1.6-21x raw, 0/16 better; dropping a quarter of the late births costs +63%), #2 plane carrier for old content (star-catch keeps only ~half of what is lost) and #3 merged births. Law: the card needs every birth's plane nearly exactly, so only NEAR-LOSSLESS compression can cut columns. OPEN: candidate 4 = exact pinned legs + unit bonds, off-diagonal s-legs in one shared top-r frame (Tucker n x r x r; r(r+1)/2 column-equivalents for ALL cohorts). Lead pre-screen on the r35 truth covariance: born-tensor energy kept at r=128 is .906/.971/.988/.993 at L4/7/10/13, at r=256 .989-.9996; covariance participation ratio 196 -> 66. Physicist runs step A (young s-projection sensitivity at r=128/192/256, kill if r=256 is worse than +5%), then step B (replaces the 384 join).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_physicist_v1/refit6b/births_prune_eval.py](code/fable_physicist_v1/refit6b/births_prune_eval.py)
- [code/fable_physicist_v1/refit6b/births_bill.py](code/fable_physicist_v1/refit6b/births_bill.py)
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
<summary>Additional programs and dependencies (17)</summary>

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
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L325](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L325)
- [TEAM_LOG_2026-09-22_to_09-26.md#L326](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L326)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-consumption-ranked-birth-pruning-dropping-quarter-late-births-raises) · [Copy provenance](../../../PROVENANCE.json)

