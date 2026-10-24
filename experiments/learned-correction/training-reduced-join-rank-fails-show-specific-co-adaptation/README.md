# Training at a reduced join rank fails to show specific co-adaptation

2026-09-25T19:46Z · experiment · rejected · measured

Additional joint training improves error at the reduced-rank schedule, but the matched full-rank control improves similarly. The isolated benefit specific to the rank cut is consistent with zero, so rank cuts must be priced at their full raw penalty.

Recorded measurement:

    - 2026-09-25 12:46 analyst: CO-ADAPTATION, loopR7s epoch 0 (VALID40, torch float64, 384/320/384 card): +1.858% raw vs loopR7 epoch 1 at all-384, against +1.934% for loopR7 epoch 1 on the same card. B vs C = -0.072% (se .070), 27/40, recovery 4%. VALID16 (physicist, MPS) recovery 1%. Epochs 1 and 2 evaluate on landing (chain 9 restarted 12:39 with an honest 5 GB peak; its earlier 'EPOCH 0 DONE' line was a kill-order artifact and is annotated).
    - 2026-09-25 12:56 analyst: CO-ADAPTATION, loopR7s epoch 1 (VALID40, torch float64, 384/320/384 card): +1.748% raw vs loopR7 epoch 1 at all-384. B vs C (loopR7 epoch 1 on the same card) = -0.179% (se .072), 25/40, recovery 9% [2, 16%] (epoch 0: 4%). VALID16 MPS: -0.206%, 15%. With the same bill, S28b (loopR7s literals) would be about -0.18% vs S28 on VALID40. Part may be the fresh cosine cycle rather than co-adaptation. Epoch 2 about 13:15.
    - 2026-09-25 13:03 lead: CO-ADAPTATION TO THE RANK CUT: NO (analyst, torch VALID40, loopR7s epoch 1 vs loopR7 epoch 1): -0.143% raw (se .047) on the all-384 card vs -0.179% (se .072) on the 384/320/384 card, so the cut-specific part is -0.036 pt (se .051), about 0. loopR7s's own cut penalty is +1.894% vs +1.934%. The penalty lives in the discarded tail directions, which no learned part can rebuild (consistent with covhunter §35/§36). loopR7s's gain is one more fresh-cosine block of general training; it still goes into S29 if its best epoch clears 0.15 pt.
    - 2026-09-25 13:03 analyst: CO-ADAPTATION ANSWER = NO (r320_chain10.sh, VALID40 torch float64). loopR7s epoch 1 vs loopR7 epoch 1: -0.143% (se .047) on the ALL-384 card and -0.179% (se .072) on the 384/320/384 card. Cut-specific part -0.036 pt (se .051) = zero. loopR7s's own cut penalty +1.894% vs +1.934%. The loopR7s gain is the extra fresh-cosine block (general), not adaptation to the cut, matching covhunter's no-stand-in law. S28b vs S28 = about -0.18% raw at the same bill.
    - 2026-09-25 13:07 [physicist] loopR7s DONE (joint, 3 epochs, trained AT 384/320/384 from loopR7 ep1; MPS): VALID16 at the schedule start 1.64623e-8 (= loopR7 ep1 no refit; +1.49% vs the same params at all-384) -> ep0 -0.012% -> ep1 1.64266e-8 -0.217% (11/16) -> ep2 1.64304e-8 -0.194%. Bar 1.63800e-8: FAILS. Co-adaptation won back ~0.21 pt of the +1.49% schedule penalty (+1.475% -> +1.267% vs all-384). CPU f64 (a) on the full S28 torch structure (s6bcard_loop5js): loopR7 ep1 all-384 1.62258e-8 (-1.565% vs S24), at 384/320/384 1.64671e-8 (+1.487%, -0.101% raw vs S24). Check (c) running.
    - 2026-09-25 13:11 [physicist] loopR7s CHECK (c) on the full S28 torch structure (s6bcard_loop5js = star-Nystrom + per-join ranks, CPU float64, VALID16, at 384/320/384): loopR7 ep1 (current S28 params) 1.64671e-8 (-0.101% vs S24; schedule cost +1.487% vs all-384); loopR7s ep1 1.64271e-8 = -0.243% vs S28 params (11/16, median -0.11%, range -1.86..+0.92%), -0.343% vs S24, schedule penalty left +1.241%; loopR7s ep2 1.64360e-8 = -0.189% (11/16). Co-adaptation to the L10 cut wins back ~0.25 pt of 1.49. Candidate for a follow-up build only (below the 0.3% literal-change bar; noisy): refit6b/s6b_loopR7s_joint_3ep/params_epoch1.pt (sha256 0a8fbdad7733a076...) + dials_epoch1.json.
    - 2026-09-25 13:12 analyst: loopR7s FINAL on VALID40 (torch float64, 384/320/384 card, vs loopR7 epoch 1 on the same card): epoch 0 -0.072% (se .070), epoch 1 -0.179% (se .072, 25/40), epoch 2 -0.154% (se .059, 26/40). Best = epoch 1, also best on VALID16. S28b with loopR7s epoch 1 is about -0.18% vs S28 at an equal bill (general training, not co-adaptation). Chains 8b/9/10 done; no analyst jobs running.
    - 2026-09-25 13:13 [physicist] REFIT LESSON recorded in memory (p2-joint-physics-dials-fit, lead's request): gains come from the number of fresh-cosine blocks, not epochs within a block (blocks -0.46/-0.47/-0.43/-0.21/-0.14%, cumulative -1.57% vs S24; epoch 1 ~= epoch 2 in each block); training AT a rank cut does not co-adapt (analyst VALID40: cut-specific -0.036 pt, se .051), so price cuts at their full raw penalty. My earlier '~1/6 recovered' note is corrected. Refit chain stopped; standing by. Checkpoint paths for the analyst/porter: refit6b/s6b_loopR7s_joint_3ep/params_epoch{0,1,2}.pt + dials_epoch{0,1,2}.json (S29 = ep1, sha256 0a8fbdad7733a076...).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_analyst_v1/r320_chain10.sh](code/fable_analyst_v1/r320_chain10.sh)
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
<summary>Additional programs and dependencies (29)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_analyst_v1/r320_features.py](code/fable_analyst_v1/r320_features.py)
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
- [code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1230](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1230)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1238](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1238)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1245](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1245)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1246](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1246)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1250](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1250)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1258](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1258)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1259](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1259)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1263](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1263)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-training-reduced-join-rank-fails-show-specific-co-adaptation) · [Copy provenance](../../../PROVENANCE.json)

