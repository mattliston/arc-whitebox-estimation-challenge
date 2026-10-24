# Transport-gate variants screened on eight networks: half-strength +0.226%, cross-form half +0.854%

2026-10-12T21:34:47Z · experiment · rejected · measured

The K4 screen (PyTorch, candidate generation 50 (S50) point, fresh A41 7100584-591, every arm paired with skew- and kurtosis-corrected transport gate (gk)'s own result on the same seed) ranked the moment-closure estimator-physics variants on top of the corrected transport gate: K31L4 age 4 at 0.6 -0.265% (7 of 8, the next lead), gk strength 1.5 -0.082% and 0.5 +0.226% (quadratic optimum about 1.9, unresolved; 1.5 vs 1.0 later null at n = 24), birth-density skew skew-corrected birth (bsk) 1.0 -0.122% (3 of 8), gate b-mode gbk 0.75 -0.073% (5 of 8), gkpB (the corrected gate also in the newborn pairs' coupling vector and beta = 2 host (1 - mean-formula derivative transport gate (gq)) in the birth leg) +0.230%, and the cross-form half xh +0.854% (1 of 8). Only the age-4 cohort and the strength went on to n = 24.

Recorded measurement:

    gk+a4k31s6 pooled -0.265%, mean log -0.252% (se 0.099), 7/8, L13-15 -0.22 / -0.18 / -0.27%; gk15 -0.082% (se 0.136, 5/8); gk05 +0.226% (se 0.130, 2/8); gk+bsk10 -0.122% (se 0.181, 3/8); gk+gb75 -0.073% (se 0.096, 5/8); gkpB +0.230% (se 0.331, 3/8); gk+xh +0.854% (se 0.360, 1/8); strength fit 0 / 0.5 / 1 / 1.5 = -0.256 / -0.476 / -0.562% mean log vs base

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/scout_card/k_table_v4.py](code/opus_reexam_20261001_v1/scout_card/k_table_v4.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/worker_sc5.py](code/opus_reexam_20261001_v1/scout_card/worker_sc5.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)

<details>
<summary>Additional programs and dependencies (40)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
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
- [code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [code/fable_porter_v1/star_nys_test.py](code/fable_porter_v1/star_nys_test.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_cut_runtime_sc1.py](code/opus_reexam_20261001_v1/scout_card/otr_cut_runtime_sc1.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1690](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1690)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1692](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1692)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1703](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1703)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1747](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1747)

<details>
<summary>All result files and records (119)</summary>

- [STOP](STOP) · result
- [chain.out](chain.out) · result
- [sc_k3a_r7100584_gk.w0.json](sc_k3a_r7100584_gk.w0.json) · result
- [sc_k4a_r7100584_gk05.w0.json](sc_k4a_r7100584_gk05.w0.json) · result
- [sc_k4a_r7100584_gk15.w0.json](sc_k4a_r7100584_gk15.w0.json) · result
- [sc_k4a_r7100585_gk05.w0.json](sc_k4a_r7100585_gk05.w0.json) · result
- [sc_k4a_r7100585_gk15.w0.json](sc_k4a_r7100585_gk15.w0.json) · result
- [sc_k4a_r7100586_gk05.w0.json](sc_k4a_r7100586_gk05.w0.json) · result
- [sc_k4a_r7100586_gk15.w0.json](sc_k4a_r7100586_gk15.w0.json) · result
- [sc_k4a_r7100587_gk05.w0.json](sc_k4a_r7100587_gk05.w0.json) · result
- [sc_k4a_r7100587_gk15.w0.json](sc_k4a_r7100587_gk15.w0.json) · result
- [sc_k4a_r7100588_gk05.w0.json](sc_k4a_r7100588_gk05.w0.json) · result
- [sc_k4a_r7100588_gk15.w0.json](sc_k4a_r7100588_gk15.w0.json) · result
- [sc_k4a_r7100589_gk05.w0.json](sc_k4a_r7100589_gk05.w0.json) · result
- [sc_k4a_r7100589_gk15.w0.json](sc_k4a_r7100589_gk15.w0.json) · result
- [sc_k4a_r7100590_gk05.w0.json](sc_k4a_r7100590_gk05.w0.json) · result
- [sc_k4a_r7100590_gk15.w0.json](sc_k4a_r7100590_gk15.w0.json) · result
- [sc_k4a_r7100591_gk05.w0.json](sc_k4a_r7100591_gk05.w0.json) · result
- [sc_k4a_r7100591_gk15.w0.json](sc_k4a_r7100591_gk15.w0.json) · result
- [sc_k4b_r7100584_gkpB.w0.json](sc_k4b_r7100584_gkpB.w0.json) · result
- [sc_k4b_r7100585_gkpB.w0.json](sc_k4b_r7100585_gkpB.w0.json) · result
- [sc_k4b_r7100586_gkpB.w0.json](sc_k4b_r7100586_gkpB.w0.json) · result
- [sc_k4b_r7100587_gkpB.w0.json](sc_k4b_r7100587_gkpB.w0.json) · result
- [sc_k4b_r7100588_gkpB.w0.json](sc_k4b_r7100588_gkpB.w0.json) · result
- [sc_k4b_r7100589_gkpB.w0.json](sc_k4b_r7100589_gkpB.w0.json) · result
- [sc_k4b_r7100590_gkpB.w0.json](sc_k4b_r7100590_gkpB.w0.json) · result
- [sc_k4b_r7100591_gkpB.w0.json](sc_k4b_r7100591_gkpB.w0.json) · result
- [sc_k4c_r7100584_gk+bsk10.w0.json](sc_k4c_r7100584_gk+bsk10.w0.json) · result
- [sc_k4c_r7100584_gk+gb75.w0.json](sc_k4c_r7100584_gk+gb75.w0.json) · result
- [sc_k4c_r7100585_gk+bsk10.w0.json](sc_k4c_r7100585_gk+bsk10.w0.json) · result
- [sc_k4c_r7100585_gk+gb75.w0.json](sc_k4c_r7100585_gk+gb75.w0.json) · result
- [sc_k4c_r7100586_gk+bsk10.w0.json](sc_k4c_r7100586_gk+bsk10.w0.json) · result
- [sc_k4c_r7100586_gk+gb75.w0.json](sc_k4c_r7100586_gk+gb75.w0.json) · result
- [sc_k4c_r7100587_gk+bsk10.w0.json](sc_k4c_r7100587_gk+bsk10.w0.json) · result
- [sc_k4c_r7100587_gk+gb75.w0.json](sc_k4c_r7100587_gk+gb75.w0.json) · result
- [sc_k4c_r7100588_gk+bsk10.w0.json](sc_k4c_r7100588_gk+bsk10.w0.json) · result
- [sc_k4c_r7100588_gk+gb75.w0.json](sc_k4c_r7100588_gk+gb75.w0.json) · result
- [sc_k4c_r7100589_gk+bsk10.w0.json](sc_k4c_r7100589_gk+bsk10.w0.json) · result
- [sc_k4c_r7100589_gk+gb75.w0.json](sc_k4c_r7100589_gk+gb75.w0.json) · result
- [sc_k4c_r7100590_gk+bsk10.w0.json](sc_k4c_r7100590_gk+bsk10.w0.json) · result
- [sc_k4c_r7100590_gk+gb75.w0.json](sc_k4c_r7100590_gk+gb75.w0.json) · result
- [sc_k4c_r7100591_gk+bsk10.w0.json](sc_k4c_r7100591_gk+bsk10.w0.json) · result
- [sc_k4c_r7100591_gk+gb75.w0.json](sc_k4c_r7100591_gk+gb75.w0.json) · result
- [sc_k4d_r7100584_gk+a4k31s6.w0.json](sc_k4d_r7100584_gk+a4k31s6.w0.json) · result
- [sc_k4d_r7100584_gk+xh.w0.json](sc_k4d_r7100584_gk+xh.w0.json) · result
- [sc_k4d_r7100585_gk+a4k31s6.w0.json](sc_k4d_r7100585_gk+a4k31s6.w0.json) · result
- [sc_k4d_r7100585_gk+xh.w0.json](sc_k4d_r7100585_gk+xh.w0.json) · result
- [sc_k4d_r7100586_gk+a4k31s6.w0.json](sc_k4d_r7100586_gk+a4k31s6.w0.json) · result
- [sc_k4d_r7100586_gk+xh.w0.json](sc_k4d_r7100586_gk+xh.w0.json) · result
- [sc_k4d_r7100587_gk+a4k31s6.w0.json](sc_k4d_r7100587_gk+a4k31s6.w0.json) · result
- [sc_k4d_r7100587_gk+xh.w0.json](sc_k4d_r7100587_gk+xh.w0.json) · result
- [sc_k4d_r7100588_gk+a4k31s6.w0.json](sc_k4d_r7100588_gk+a4k31s6.w0.json) · result
- [sc_k4d_r7100588_gk+xh.w0.json](sc_k4d_r7100588_gk+xh.w0.json) · result
- [sc_k4d_r7100589_gk+a4k31s6.w0.json](sc_k4d_r7100589_gk+a4k31s6.w0.json) · result
- [sc_k4d_r7100589_gk+xh.w0.json](sc_k4d_r7100589_gk+xh.w0.json) · result
- [sc_k4d_r7100590_gk+a4k31s6.w0.json](sc_k4d_r7100590_gk+a4k31s6.w0.json) · result
- [sc_k4d_r7100590_gk+xh.w0.json](sc_k4d_r7100590_gk+xh.w0.json) · result
- [sc_k4d_r7100591_gk+a4k31s6.w0.json](sc_k4d_r7100591_gk+a4k31s6.w0.json) · result
- [sc_k4d_r7100591_gk+xh.w0.json](sc_k4d_r7100591_gk+xh.w0.json) · result
- [guard_w0_20261012T190921Z.json](guard_w0_20261012T190921Z.json) · result
- [sc_k3a_r7100584_gk.json](sc_k3a_r7100584_gk.json) · result
- [sc_k4a_r7100584_gk05.json](sc_k4a_r7100584_gk05.json) · result
- [sc_k4a_r7100584_gk15.json](sc_k4a_r7100584_gk15.json) · result
- [sc_k4a_r7100585_gk05.json](sc_k4a_r7100585_gk05.json) · result
- [sc_k4a_r7100585_gk15.json](sc_k4a_r7100585_gk15.json) · result
- [sc_k4a_r7100586_gk05.json](sc_k4a_r7100586_gk05.json) · result
- [sc_k4a_r7100586_gk15.json](sc_k4a_r7100586_gk15.json) · result
- [sc_k4a_r7100587_gk05.json](sc_k4a_r7100587_gk05.json) · result
- [sc_k4a_r7100587_gk15.json](sc_k4a_r7100587_gk15.json) · result
- [sc_k4a_r7100588_gk05.json](sc_k4a_r7100588_gk05.json) · result
- [sc_k4a_r7100588_gk15.json](sc_k4a_r7100588_gk15.json) · result
- [sc_k4a_r7100589_gk05.json](sc_k4a_r7100589_gk05.json) · result
- [sc_k4a_r7100589_gk15.json](sc_k4a_r7100589_gk15.json) · result
- [sc_k4a_r7100590_gk05.json](sc_k4a_r7100590_gk05.json) · result
- [sc_k4a_r7100590_gk15.json](sc_k4a_r7100590_gk15.json) · result
- [sc_k4a_r7100591_gk05.json](sc_k4a_r7100591_gk05.json) · result
- [sc_k4a_r7100591_gk15.json](sc_k4a_r7100591_gk15.json) · result
- [sc_k4b_r7100584_gkpB.json](sc_k4b_r7100584_gkpB.json) · result
- [sc_k4b_r7100585_gkpB.json](sc_k4b_r7100585_gkpB.json) · result
- [sc_k4b_r7100586_gkpB.json](sc_k4b_r7100586_gkpB.json) · result
- [sc_k4b_r7100587_gkpB.json](sc_k4b_r7100587_gkpB.json) · result
- [sc_k4b_r7100588_gkpB.json](sc_k4b_r7100588_gkpB.json) · result
- [sc_k4b_r7100589_gkpB.json](sc_k4b_r7100589_gkpB.json) · result
- [sc_k4b_r7100590_gkpB.json](sc_k4b_r7100590_gkpB.json) · result
- [sc_k4b_r7100591_gkpB.json](sc_k4b_r7100591_gkpB.json) · result
- [sc_k4c_r7100584_gk+bsk10.json](sc_k4c_r7100584_gk+bsk10.json) · result
- [sc_k4c_r7100584_gk+gb75.json](sc_k4c_r7100584_gk+gb75.json) · result
- [sc_k4c_r7100585_gk+bsk10.json](sc_k4c_r7100585_gk+bsk10.json) · result
- [sc_k4c_r7100585_gk+gb75.json](sc_k4c_r7100585_gk+gb75.json) · result
- [sc_k4c_r7100586_gk+bsk10.json](sc_k4c_r7100586_gk+bsk10.json) · result
- [sc_k4c_r7100586_gk+gb75.json](sc_k4c_r7100586_gk+gb75.json) · result
- [sc_k4c_r7100587_gk+bsk10.json](sc_k4c_r7100587_gk+bsk10.json) · result
- [sc_k4c_r7100587_gk+gb75.json](sc_k4c_r7100587_gk+gb75.json) · result
- [sc_k4c_r7100588_gk+bsk10.json](sc_k4c_r7100588_gk+bsk10.json) · result
- [sc_k4c_r7100588_gk+gb75.json](sc_k4c_r7100588_gk+gb75.json) · result
- [sc_k4c_r7100589_gk+bsk10.json](sc_k4c_r7100589_gk+bsk10.json) · result
- [sc_k4c_r7100589_gk+gb75.json](sc_k4c_r7100589_gk+gb75.json) · result
- [sc_k4c_r7100590_gk+bsk10.json](sc_k4c_r7100590_gk+bsk10.json) · result
- [sc_k4c_r7100590_gk+gb75.json](sc_k4c_r7100590_gk+gb75.json) · result
- [sc_k4c_r7100591_gk+bsk10.json](sc_k4c_r7100591_gk+bsk10.json) · result
- [sc_k4c_r7100591_gk+gb75.json](sc_k4c_r7100591_gk+gb75.json) · result
- [sc_k4d_r7100584_gk+a4k31s6.json](sc_k4d_r7100584_gk+a4k31s6.json) · result
- [sc_k4d_r7100584_gk+xh.json](sc_k4d_r7100584_gk+xh.json) · result
- [sc_k4d_r7100585_gk+a4k31s6.json](sc_k4d_r7100585_gk+a4k31s6.json) · result
- [sc_k4d_r7100585_gk+xh.json](sc_k4d_r7100585_gk+xh.json) · result
- [sc_k4d_r7100586_gk+a4k31s6.json](sc_k4d_r7100586_gk+a4k31s6.json) · result
- [sc_k4d_r7100586_gk+xh.json](sc_k4d_r7100586_gk+xh.json) · result
- [sc_k4d_r7100587_gk+a4k31s6.json](sc_k4d_r7100587_gk+a4k31s6.json) · result
- [sc_k4d_r7100587_gk+xh.json](sc_k4d_r7100587_gk+xh.json) · result
- [sc_k4d_r7100588_gk+a4k31s6.json](sc_k4d_r7100588_gk+a4k31s6.json) · result
- [sc_k4d_r7100588_gk+xh.json](sc_k4d_r7100588_gk+xh.json) · result
- [sc_k4d_r7100589_gk+a4k31s6.json](sc_k4d_r7100589_gk+a4k31s6.json) · result
- [sc_k4d_r7100589_gk+xh.json](sc_k4d_r7100589_gk+xh.json) · result
- [sc_k4d_r7100590_gk+a4k31s6.json](sc_k4d_r7100590_gk+a4k31s6.json) · result
- [sc_k4d_r7100590_gk+xh.json](sc_k4d_r7100590_gk+xh.json) · result
- [sc_k4d_r7100591_gk+a4k31s6.json](sc_k4d_r7100591_gk+a4k31s6.json) · result
- [sc_k4d_r7100591_gk+xh.json](sc_k4d_r7100591_gk+xh.json) · result
- [worker_w0.log](worker_w0.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-transport-gate-variants-screened-eight-networks-half-strength-0) · [Copy provenance](../../../PROVENANCE.json)

