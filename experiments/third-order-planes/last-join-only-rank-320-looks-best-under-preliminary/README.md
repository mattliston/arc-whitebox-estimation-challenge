# Last-join-only rank 320 looks best under the preliminary cost model, later contradicted by native bills

2026-09-25T17:13Z · experiment · informational · measured

Reducing only the layer-13 join limits downstream damage while capturing much of the rank-dependent work saving. Earlier rank cuts and rank 256 were worse, so the final decision awaited native held-out results.

Recorded measurement:

    - 2026-09-25 10:13 lead: RANK GRID (analyst, torch, VALID40, per join L7/L10/L13): 384/384/320 (L13 only) adjusted -0.832% (30/40; raw +1.07% se .17, bill -1.87%) beats 384/320/320 (-0.588%, 24/40) and all-320 (-0.183%). L10 only +0.305% and L7 only +0.444% are harmful. 256 at L13 is +0.707%, because the cost is convex in rank. The joins add linearly (raw +1.82 + 1.07 = 2.89 vs 2.88). Only the last join has a good exchange rate: 43% of the rank-dependent bill, and its damage is limited to L13-15.
      DECISION RULE, fixed before the held-out numbers (~10:34): ship L13-only unless 384/320/320 beats it by more than 0.2 pt adjusted on the 80-net torch pool. Porter builds S27s = {7:384, 10:384, 13:320}; the final S28 schedule is chosen on native 80-net pooled adjusted (S27r and S27s, each on VALID40 plus the held-out 40). loopR7s trains on the chosen schedule, 384/384/320 by default at 10:40.
    - 2026-09-25 10:33 analyst: R320 SCHEDULE RESULT (fable_analyst_v1/RESULT.md section 43). Best fixed schedule = old-tier rank 320 at the L13 JOIN ONLY (L7, L10 at 384): VALID40 raw +1.07% (se .17), bill -1.87%, adjusted -0.832%, 30/40 better (torch, native-calibrated per net). L10-only +0.305% and L7-only +0.444% are net harmful; 256 at L13 +0.707% is too deep; 384/320/320 -0.588%; S25's all-320 -0.183%. The joins add up (L10 +1.82 + L13 +1.07 = 2.89 vs 2.88). No router generalizes: VALID40-fitted resp_L10 rules lose to always-on on 40 held-out TRAIN640 nets. Held-out rows for L13-only are queued behind the memory gate. Suggested build: R320 at L13 only on the current stack, bar about adjusted <= -0.5%.
    - 2026-09-25 10:35 lead: SCHEDULE PICK = 384/384/320 (320 at the L13 join only), per the rule fixed at 10:08. Analyst (torch): VALID40 adjusted -0.832% [-1.15, -0.52] 30/40; estimated 80-net pool about -0.96% vs 384/320/320's measured -0.987% [-1.43, -0.54] (difference well under 0.2 pt). Provisional until L13-only's own held-out torch rows run (blocked by memory since 10:12); the pick would flip only if its held-out raw penalty reached +1.16% or more (VALID40 +1.07%; held-out penalties ran 25-30% below VALID40 for every schedule measured). loopR7s trains at 384/384/320, launched after S27s's smoke or at 11:00. S28 default = S27s's schedule; the native 80-net comparison (porter) decides in the end.
    - 2026-09-25 10:36 analyst: R320 SCHEDULE PICK (provisional) = 384/384/320 (rank 320 at the L13 join only), sent to lead, porter and physicist (loopR7s). VALID40 -0.832% [95% -1.15, -0.52], 30/40. 384/320/320: VALID40 -0.588% [-1.23, +0.07]; HELD-OUT 40 -1.373% [-1.96, -0.78] 32/40 (passes the lead's held-out bar <= -0.30%, CI excl 0); pooled 80 -0.987% [-1.43, -0.54]. L13-only held-out estimated -1.09% (pooled 80 about -0.96%) from VALID40 per-join additivity. By the lead's rule (L13-only unless 384/320/320 is > 0.2 pt better on 80 nets) it flips only if its held-out comes out worse than -0.73%. Its held-out rows are blocked by the memory gate since 10:12; the half-cut grid (384/352/320, 352/352/320, 384/384/352, 384/384/288) is queued behind them.

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
<summary>Additional programs and dependencies (25)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_analyst_v1/r320_features.py](code/fable_analyst_v1/r320_features.py)
- [code/fable_analyst_v1/r320_perjoin.py](code/fable_analyst_v1/r320_perjoin.py)
- [code/fable_analyst_v1/r320_schedules_v1.py](code/fable_analyst_v1/r320_schedules_v1.py)
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

- [TEAM_LOG_2026-09-22_to_09-26.md#L1079](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1079)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1080](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1080)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1086](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1086)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1087](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1087)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1089](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1089)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT-1a40d4c3.md](../gain-mixture-correction-half-strength-native-adjusted-ratio-1/RESULT-1a40d4c3.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-last-join-only-rank-320-looks-best-under-preliminary) · [Copy provenance](../../../PROVENANCE.json)

