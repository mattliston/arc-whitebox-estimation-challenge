# Twice-wide learned trunk and fourfold training data fail to beat the previous fit materially

2026-09-24T20:40Z · experiment · rejected · measured

Function-preserving width expansion produced no useful held-out gain. The larger-data trunk and trainable-knob runs also gave at most a small change relative to the previous trained estimator.

Recorded measurement:

    2026-09-24 13:40 covhunter: CAPACITY TEST LAUNCHED (lead): fable_covhunter_v1/wide2x_B2recipe_1ep (log wide2x_B2recipe_1ep.log). GNN trunk hidden widths 2x (enc0 126->256, enc2 256->256, enc4 256->64; msg cm16/state 64/heads untouched; trunk 44,144 -> 117,872 params), function-preserving init from B2 ep1 (make_wide_init.py: new rows N(0,std), zero output blocks; s6bcard_wide.py payload). Certificate net 37 @5 thr: wide == B2 ep1 BIT-EXACT at step 0; backward finite, zero blocks get grads, new input rows exactly 0 at step 0. Run: B2 recipe (full+P, lr 1e-4/5e-5, warmup 3/5, clip 1), 1 epoch TRAIN160, VALID16, 5 thr nice 10. Bar: VALID16 <= -1.0% vs B2 ep1 1.74614e-8 (<= 1.72868e-8). Caveat: control = B2 ep1 without the extra epoch. ETA ~2 h. R320 closed (RESULT sec 21).
    - 2026-09-24 13:55 lead: covhunter capacity test running: 2x-wide GNN trunk (44k -> 118k params), bit-exact to B2 ep1 at step 0, 1 epoch B2 recipe, bar VALID16 <= 1.72868e-8 (-1%), ETA ~15:45.
    - 2026-09-24 16:10 lead: CAPACITY CLOSED. covhunter wide2x (2x-wide GNN trunk, 1 epoch B2 recipe): VALID16 1.74850e-8 vs B2 ep1 1.74614e-8 (+0.14%). T640 TRUNK DONE: VALID40 1.79637e-8 vs B2 ep1 1.79867e-8 (-0.13%, within noise; vs B2 ep0 -0.36%). The analyst gives the per-net comparison. Learned parts are neither data- nor capacity-limited. j05 refit (physicist, started 15:50): valid_before (j1 X=.5 at B2 ep1 literals, learned ON) = 1.8329e-8 (+4.97% vs B2 ep1). The bar is 1.7374e-8, so the refit must recover ~5.5 pt. T640 knobs still running (~19:05).
    - 2026-09-24 17:16 analyst: T640 ENDS: full+trunk VALID40 1.796375e-8 = -1.45% vs before (37/40, PASS), -0.13% vs B2 ep1 (24/40; VALID16 tie); full+knobs -0.73% (33/40, PASS) but +0.60% vs B2 ep1; knobs barely moved. 4x data + knobs ~ +0.1%: trunk lever saturating. Strategic top-3 sent (final-layer-born error test; LS-refit b; top-32 C15 subspace correction; crazy: knob hypernetwork; f16 dead: billed like f32). RESULT.md sec 31.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_covhunter_v1/make_wide_init.py](code/fable_covhunter_v1/make_wide_init.py)
- [code/fable_covhunter_v1/s6bcard_wide.py](code/fable_covhunter_v1/s6bcard_wide.py)
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

- [TEAM_LOG_2026-09-22_to_09-26.md#L396](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L396)
- [TEAM_LOG_2026-09-22_to_09-26.md#L399](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L399)
- [TEAM_LOG_2026-09-22_to_09-26.md#L458](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L458)
- [TEAM_LOG_2026-09-22_to_09-26.md#L476](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L476)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [wide2x_B2recipe_1ep.log](wide2x_B2recipe_1ep.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-twice-wide-learned-trunk-fourfold-training-data-fail-beat) · [Copy provenance](../../../PROVENANCE.json)

