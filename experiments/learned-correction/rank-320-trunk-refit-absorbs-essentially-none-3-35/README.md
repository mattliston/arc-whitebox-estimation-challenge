# Rank-320 trunk refit absorbs essentially none of the 3.35% raw-error loss

2026-09-24T17:15Z · experiment · rejected · measured

The rank-reduced estimator was warm-started from the trained full-rank model and refit with the same recipe. Held-out error stayed almost unchanged and missed break-even.

Recorded measurement:

    2026-09-24 10:15 covhunter: TIER-CUT + TRUNK REFIT test LAUNCHED (lead): fable_covhunter_v1/r320_B2recipe_1ep (log r320_B2recipe_1ep.log). Plumbing = s6bcard_r320.py (sets covstep.TIER_RANK, the global the HookCard passes to the every3 push; nothing else depends on rank). Certificate: RANK 384 wrapper == s6bcard bit-exact; old-tier Q is 1024x320 from L8 on; no-refit r37 with B2-ep1 params +7.93% at 320. Run: refit6b_aux.py --card s6bcard_r320 --aux_w 0 (= B2's exact loss) with B2's recipe (full+P weights-only, lr 1e-4/5e-5, warmup 3/5, clip 1), 1 epoch TRAIN160, warm start B2 ep1, VALID16, 5 threads nice 10. Bar: VALID16 <= 1.78106e-8 (B2 ep1 at 384 1.74614e-8 +2%). ETA ~11:25.
    - 2026-09-24 11:56 lead: R320 + B2-recipe refit (covhunter; 1 epoch TRAIN160, warm start B2 ep1): VALID16 1.80453e-8 vs 1.80469e-8 before the refit, so the refit absorbed ~0% of the +3.35% rank cut (bar 1.78106e-8): FAIL. R320 CLOSED. Law: structural information cuts are NOT absorbed by trunk refits, consistent with the physicist's learned-OFF test. S12 falls back to S12-lite = S11 + colpack + deadcut35 (the porter's stage runs its VALID40 automatically; expected ~-0.7% adjusted). RESUMED the T640 knobs refit (kill -CONT 69931) now that r320 has freed CPU/memory (74% free, load 6.5).
    2026-09-24 11:55 porter: covhunter r320 VALID16 1.804525994584575e-08 (bar <= 1.78106e-8, pass=0; checks ok=1) -> S12-lite (S11 + colpack + deadcut35) VALID40 vs S11 rows STARTED.
    2026-09-24 12:06 porter: R320 OUT (covhunter VALID16 1.8045e-8 > 1.78106e-8; native r100 still +10.65% raw after refit). S12-lite (S11+colpack+deadcut35) VALID40 vs S11 running (~12:20). S12-lite+read6q built (c20d67c7; read6q last; gate + validate OK) and proposed as the next VALID40 / fire candidate.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_covhunter_v1/s6bcard_r320.py](code/fable_covhunter_v1/s6bcard_r320.py)
- [code/fable_covhunter_v1/refit6b_aux.py](code/fable_covhunter_v1/refit6b_aux.py)
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

- [TEAM_LOG_2026-09-22_to_09-26.md#L315](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L315)
- [TEAM_LOG_2026-09-22_to_09-26.md#L367](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L367)
- [TEAM_LOG_2026-09-22_to_09-26.md#L369](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L369)
- [TEAM_LOG_2026-09-22_to_09-26.md#L375](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L375)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [r320_B2recipe_1ep.log](r320_B2recipe_1ep.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-rank-320-trunk-refit-absorbs-essentially-none-3-35) · [Copy provenance](../../../PROVENANCE.json)

