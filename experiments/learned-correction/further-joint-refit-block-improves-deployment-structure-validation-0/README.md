# A further joint-refit block improves deployment-structure validation 0.430% but is superseded before submission

2026-09-25T17:42Z · experiment · informational · measured

This intermediate joint-refit block improves the prior checkpoint on deployment-structure CPU validation. It becomes a candidate base before a later block supersedes it in the submitted parameters.

Recorded measurement:

    - 2026-09-25 10:42 [physicist] loopR6 DONE: VALID16 (MPS) start 1.63206e-8 -> ep0 1.62885e-8 (-0.196%, 11/16) -> ep1 1.62581e-8 (-0.383%, 13/16, median -0.43%) -> ep2 1.62604e-8 (-0.369%). Bar 1.62390e-8: FAILS (best -0.383%). Diminishing per block: arm A -0.46%, loopR5 -0.47%, loopR6 -0.38%; cumulative vs S24 params on VALID16 (MPS) -1.32% at loopR6 ep1. No overfit (train 1.661 -> 1.656e-8). CPU float64 check of R6ep1/ep2 on the S24-structure card launched for the S29-vs-S28 call.
    - 2026-09-25 10:43 lead: loopR6 DONE (physicist, MPS VALID16): best epoch 1 -0.383% vs its start (13/16), epoch 2 -0.369%; misses its pre-registered bar (1.62390e-8). Blocks are shrinking (-0.46, -0.47, -0.38%); no overfit (train 1.656e-8). Cumulative vs S24's params about -1.32% (VALID16). CPU f64 check vs loopR5 epoch 2 running.
    - 2026-09-25 10:45 [physicist] loopR6 on the DEPLOYMENT structure (no refit, VALID16, CPU float64, S24-structure card s6bcard_loop5j): ep1 1.62592e-8 = -1.362% vs S24, -0.430% vs loopR5 ep2 (14/16, median -0.37%, range -1.04..+0.09%); ep2 1.62608e-8 = -1.352% vs S24, -0.420% vs loopR5 ep2 (14/16). S29 CANDIDATE (lead's call; judge = native VALID40 vs S28): S27 structure + refit6b/s6b_loopR6_joint_3ep/params_epoch1.pt (sha256 ef2e4e73fec25de8...) + dials_epoch1.json ({'X1': 0.5558, 'X0': -0.0003, 'X2': 0.365, 'X3': 0.1515, 'bscale': 1.0398, 's_carry': 1.0032, 's_new': 0.9942, 'c0': 0.599, 'c1': 0.5027, 'k4a': 0.9841, 'k4b': 0.967, 'k4c': 0.9608}).
    - 2026-09-25 10:46 lead: S27r (384/320/320) NATIVE VALID40 vs S27 rows: raw +2.917% [+2.23, +3.60], util -3.167% [-3.185, -3.147], adjusted -0.343% [-1.01, +0.32], 23/40 -> FAIL on VALID40 alone. The raw penalty matches torch (+2.88%); the native bill cut is 6% smaller than the analyst's unit-cost model (-3.17% vs -3.38%). The 80-net native pool (held-out arms) still decides. For L13-only (S27s), scaling that bill shortfall suggests about -1.75% bill and about -0.7% adjusted on VALID40.
      loopR6 epoch 1 on the deployment card (physicist, CPU f64 VALID16): -0.430% vs loopR5 epoch 2 (14/16), -1.362% vs S24. It becomes S28's params (params_epoch1.pt ef2e4e73 + dials_epoch1.json; c0 .599).

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
<summary>Additional programs and dependencies (27)</summary>

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
- [code/fable_engineer_v1/mps_card.py](code/fable_engineer_v1/mps_card.py)
- [code/fable_engineer_v1/mps_lock.py](code/fable_engineer_v1/mps_lock.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/refit_mps_v3j2.py](code/fable_physicist_v1/refit6b/refit_mps_v3j2.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop4j.py](code/fable_physicist_v1/refit6b/s6bcard_loop4j.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop5j.py](code/fable_physicist_v1/refit6b/s6bcard_loop5j.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1093](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1093)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1094](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1094)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1096](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1096)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1098](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1098)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1099](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1099)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- enables: [Final plain joint-refit block improves deployment validation 0.206% and is selected before rank adaptation](../final-plain-joint-refit-block-improves-deployment-validation-0/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-further-joint-refit-block-improves-deployment-structure-validation-0) · [Copy provenance](../../../PROVENANCE.json)

