# Increasing the final join rank beyond 384 costs more than its accuracy gain

2026-09-25T19:20Z · experiment · rejected · measured

Increasing the layer-13 join rank to 448 or 512 improves raw error, but the additional arithmetic makes both adjusted results worse. Rank 384 is retained.

Recorded measurement:

    2026-09-25 12:20 porter: BILL MAP RAISE ARMS (native census nets 35/100, mean util vs S27, no setup): 384/320/448 -1.764%, 384/320/512 -1.058% (S27t 384/320/384 -2.483%) = +0.737% / +1.461% util vs S27t; L14/L15 core verified 448/512, envelope width 448.0/512.0; extra cost at the L13 join push (+0.60 pts of S27 total at 512 on r35). Sent to lead + analyst (stack/billmap/REPORT_v3.txt).
    2026-09-25 12:27 porter: analyst priced the raise arms with the native bills + torch raw at S28 params (VALID40): 384/320/448 adj -0.004% vs S27 (+0.57 pt worse than 384/320/384), 384/320/512 +0.667% (+1.24 pt worse). Raise line CLOSED; S28 keeps 384/320/384.
    - 2026-09-25 12:28 analyst: RAISE ARMS LOSE (RESULT section 45; VALID40 torch raw at the S28 params x the porter's REPORT_v3 native bills): 384/320/448 adjusted -0.004%, which is +0.566 pt [+0.45, +0.68] worse than 384/320/384; 384/320/512 +0.667%, +1.237 pt [+1.13, +1.34] worse. A 64-rank raise at L13 costs +0.72% of the bill for -0.16% of raw (se .06). At the S28 params the schedule is -0.570% [-1.12, +0.01] on VALID40 (raw +1.934%). Chain 9 waits for loopR7s epoch 0.
    - 2026-09-25 12:28 lead: L13 RAISE CLOSED (analyst torch raw at loopR7 epoch 1 x porter native bills, REPORT_v3): 384/320/448 raw +1.766% vs +1.934% (only -0.17 pt), bill +0.737% vs S27t, so adjusted +0.566 pt WORSE than 384/320/384; 512 is +1.237 pt worse. Each 64 ranks at L13 costs about +0.72% bill for about -0.16% raw. 384 sits at the knee: going down to 320 costs +1.07% raw, going up to 448 buys back only 0.16%. At S28 params the schedule alone is about -0.57% on VALID40 (held-out about -1.2%).
    - 2026-09-25 13:18 analyst: L13 RAISE TASK DONE, ALL LOSE (RESULT section 45; VALID40 torch at the S28 params x native bills vs S27t): 384/320/448 raw -0.164% (se .058) for +0.737% bill = +0.57% adjusted; 384/320/512 raw -0.213% for +1.461% = +1.24%; reference 384/384/448 raw -0.165% vs all-384 (the gain is independent of the L10 rank). Cause: at the L13 join the tail beyond 384 holds only 22% of the 320-384 slice's energy (missed energy 9.1e-5 at 320, 1.65e-5 at 384). At the cut's 0.14% raw per 1e-5, the whole tail is worth at most about 0.24% raw, while each 64 ranks costs about +0.74% of the bill. Pick stays 384/320/384; no held-out. Card clean: intended ranks on all 40 nets, L13 live 768-864 > 512. The 448/512 bills are linear (no padding). The overflow envelope is native-only (porter widened it).

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
<summary>Additional programs and dependencies (28)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_analyst_v1/r320_features.py](code/fable_analyst_v1/r320_features.py)
- [code/fable_analyst_v1/r320_perjoin.py](code/fable_analyst_v1/r320_perjoin.py)
- [code/fable_analyst_v1/r320_schedules.py](code/fable_analyst_v1/r320_schedules.py)
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
- [code/fable_porter_v1/census_boundary.py](code/fable_porter_v1/census_boundary.py)
- [code/fable_porter_v1/stack/billmap/pre_384-320-448.py](code/fable_porter_v1/stack/billmap/pre_384-320-448.py)
- [code/fable_porter_v1/stack/billmap/pre_384-320-512.py](code/fable_porter_v1/stack/billmap/pre_384-320-512.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1216](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1216)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1217](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1217)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1218](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1218)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1219](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1219)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1274](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1274)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [REPORT_v3.txt](REPORT_v3.txt) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-increasing-final-join-rank-beyond-384-costs-more-than) · [Copy provenance](../../../PROVENANCE.json)

