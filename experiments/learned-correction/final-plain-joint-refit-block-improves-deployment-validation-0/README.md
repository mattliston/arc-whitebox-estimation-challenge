# Final plain joint-refit block improves deployment validation 0.206% and is selected before rank adaptation

2026-09-25T18:04Z · experiment · adopted · measured

Another block showed diminishing gains and completed before its stop instruction arrived. Its best checkpoint passed CPU deployment-structure comparison and became the pre-adaptation parameter base.

Recorded measurement:

    - 2026-09-25 11:04 lead: PLAIN JOINT REFIT IS SPENT. loopR7 (plain continuation from loopR6 epoch 1, launched 10:46 before my redirect) epoch 0: VALID16 -0.011% vs start (9/16), against -0.17/-0.20% at the same point for loopR5/R6. STOPPED to free the GPU and memory for the S28 path. Next GPU job: loopR7s at 384/384/320 (co-adaptation to the L13 cut), after the per-join rank parity check and S27s's smoke. S28 params stay loopR6 epoch 1.
    - 2026-09-25 11:40 [physicist] loopR7 DONE: VALID16 (MPS) start 1.62581e-8 -> ep0 1.62564e-8 (-0.011%) -> ep1 1.62210e-8 (-0.228%, 14/16) -> ep2 1.62204e-8 (-0.232%, 14/16, median -0.16%, range -0.67..+0.07%). Bar 1.61768e-8: FAILS. The optimization-length lever has flattened (blocks: arm A -0.46%, loopR5 -0.47%, loopR6 -0.38%, loopR7 -0.23%); cumulative vs S24 params (MPS VALID16) -1.55%. No overfit (train 1.656 -> 1.648e-8). CPU float64 check of R7 ep1/ep2 on the S24-structure card launched (S30 vs S29 call).
    - 2026-09-25 11:43 [physicist] loopR7 on the DEPLOYMENT structure (no refit, VALID16, CPU float64, S24-structure card): ep1 1.62256e-8 = -1.565% vs S24, -0.206% vs loopR6 ep1 (13/16, median -0.22%, range -0.49..+0.15%); ep2 1.62274e-8 = -1.555% vs S24, -0.195% vs loopR6 ep1 (12/16). FINAL CANDIDATE of the chain (S30, lead's call; judge = native VALID40): S27 structure + refit6b/s6b_loopR7_joint_3ep/params_epoch1.pt (sha256 1e16861a26d66e6c...) + dials_epoch1.json ({'X1': 0.5551, 'X0': -0.0023, 'X2': 0.3768, 'X3': 0.1515, 'bscale': 1.0381, 's_carry': 1.0033, 's_new': 0.9941, 'c0': 0.5712, 'c1': 0.5027, 'k4a': 0.9839, 'k4b': 0.966, 'k4c': 0.9577}). Per-block gains on the S24 card: arm A -0.465%, R5 -0.47%, R6 -0.43%, R7 -0.21% -> recommend STOPPING the chain (next block ~-0.1%).
    - 2026-09-25 11:44 lead: loopR7 (plain, finished before my stop reached the physicist) on the deployment card, CPU f64 VALID16: epoch 1 -0.21% vs loopR6 epoch 1 (13/16), -1.565% vs S24; epoch 2 -1.555%. Blocks halved (-0.47, -0.43, -0.21%); plain refit ends here. S28 params = loopR7 epoch 1 (params_epoch1.pt 1e16861a + dials_epoch1.json; c0 .571). NEXT GPU job: loopR7s from loopR7 epoch 1, trained AT 384/320/384 (co-adaptation to the L10 cut, raw +1.82% on VALID40), per-join parity first. If it passes its CPU check by about 13:15, the porter builds S28b with its literals; both go through VALID40, and the better one fires at 17:00.

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

- [TEAM_LOG_2026-09-22_to_09-26.md#L1104](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1104)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1174](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1174)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1180](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1180)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1181](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1181)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332361: middle join rank 320 with refitted learned correction and matched dials](../../../submissions/332361/README.md)
- ← enables: [A further joint-refit block improves deployment-structure validation 0.430% but is superseded before submission](../further-joint-refit-block-improves-deployment-structure-validation-0/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-final-plain-joint-refit-block-improves-deployment-validation-0) · [Copy provenance](../../../PROVENANCE.json)

