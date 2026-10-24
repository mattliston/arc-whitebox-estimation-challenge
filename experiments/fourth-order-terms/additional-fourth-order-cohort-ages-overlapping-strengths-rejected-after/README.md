# Additional fourth-order cohort ages and overlapping strengths rejected after the layers 8-15 quartic extension

2026-10-13T16:21Z · experiment · rejected · measured

With cohort-age local fourth-order term (K4L) ages 3-4 on, the remaining age caps of the fourth-order carriers add nothing: K4L ages 5-6 where unjoined, F4C age 3 at half strength, cohort-age (3,1) plane term (K31L) age 4 at L8-11; scaling K31L4, F4C or K4L ages 1-2 down shows no duplication; the per-unit k4 residual is not in the sources the moment-closure estimator leaves out.

Recorded measurement:

    ages 5-6 -0.028% (se 0.096); F4C age 3 half -0.027%; K31L age 4 L8-11 -0.075% (1.1 se); k31l x0.75 +0.059%, f4c x0.85 +0.054%, k4l x0.9 -0.084%; residual sources explain 5.3% LOO (bar 20%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/physicist/t144_table.py](code/opus_reexam_20261001_v1/physicist/t144_table.py)
- [code/opus_reexam_20261001_v1/physicist/t143_k4_residual.py](code/opus_reexam_20261001_v1/physicist/t143_k4_residual.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)

<details>
<summary>Additional programs and dependencies (41)</summary>

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
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_cut_runtime_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_cut_runtime_sc4.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc7.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc7.py)
- [code/opus_reexam_20261001_v1/scout_card/worker_sc11.py](code/opus_reexam_20261001_v1/scout_card/worker_sc11.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L2071](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2071)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2103](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2103)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2125](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2125)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2127](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2127)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2135](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2135)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2136](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2136)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2139](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2139)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2140](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2140)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2142](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2142)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2162](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2162)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2164](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2164)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2165](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2165)

<details>
<summary>All result files and records (93)</summary>

- [sc_k230_r7100584_cbase.w0.json](sc_k230_r7100584_cbase.w0.json) · result
- [sc_k231_r7100584_c_a4k31e.w0.json](sc_k231_r7100584_c_a4k31e.w0.json) · result
- [sc_k231_r7100585_c_a4k31e.w0.json](sc_k231_r7100585_c_a4k31e.w0.json) · result
- [sc_k231_r7100586_c_a4k31e.w0.json](sc_k231_r7100586_c_a4k31e.w0.json) · result
- [sc_k231_r7100587_c_a4k31e.w0.json](sc_k231_r7100587_c_a4k31e.w0.json) · result
- [sc_k231_r7100588_c_a4k31e.w0.json](sc_k231_r7100588_c_a4k31e.w0.json) · result
- [sc_k231_r7100589_c_a4k31e.w0.json](sc_k231_r7100589_c_a4k31e.w0.json) · result
- [sc_k231_r7100590_c_a4k31e.w0.json](sc_k231_r7100590_c_a4k31e.w0.json) · result
- [sc_k231_r7100591_c_a4k31e.w0.json](sc_k231_r7100591_c_a4k31e.w0.json) · result
- [guard_v3_w0_20261013T144200Z.json](guard_v3_w0_20261013T144200Z.json) · result
- [sc_k230_r7100584_cbase.json](sc_k230_r7100584_cbase.json) · result
- [sc_k231_r7100584_c_a4k31e.json](sc_k231_r7100584_c_a4k31e.json) · result
- [sc_k231_r7100585_c_a4k31e.json](sc_k231_r7100585_c_a4k31e.json) · result
- [sc_k231_r7100586_c_a4k31e.json](sc_k231_r7100586_c_a4k31e.json) · result
- [sc_k231_r7100587_c_a4k31e.json](sc_k231_r7100587_c_a4k31e.json) · result
- [sc_k231_r7100588_c_a4k31e.json](sc_k231_r7100588_c_a4k31e.json) · result
- [sc_k231_r7100589_c_a4k31e.json](sc_k231_r7100589_c_a4k31e.json) · result
- [sc_k231_r7100590_c_a4k31e.json](sc_k231_r7100590_c_a4k31e.json) · result
- [sc_k231_r7100591_c_a4k31e.json](sc_k231_r7100591_c_a4k31e.json) · result
- [sc_k231_r7100584_c_a4k31.json](sc_k231_r7100584_c_a4k31.json) · result
- [sc_k231_r7100585_c_a4k31.json](sc_k231_r7100585_c_a4k31.json) · result
- [sc_k231_r7100586_c_a4k31.json](sc_k231_r7100586_c_a4k31.json) · result
- [sc_k231_r7100587_c_a4k31.json](sc_k231_r7100587_c_a4k31.json) · result
- [sc_k231_r7100588_c_a4k31.json](sc_k231_r7100588_c_a4k31.json) · result
- [sc_k231_r7100589_c_a4k31.json](sc_k231_r7100589_c_a4k31.json) · result
- [sc_k231_r7100590_c_a4k31.json](sc_k231_r7100590_c_a4k31.json) · result
- [sc_k231_r7100591_c_a4k31.json](sc_k231_r7100591_c_a4k31.json) · result
- [worker_w0.log](worker_w0.log) · log
- [sc_k240_r7100584_cbase.w0.json](sc_k240_r7100584_cbase.w0.json) · result
- [sc_k241_r7100584_c_f4c85.w1.json](sc_k241_r7100584_c_f4c85.w1.json) · result
- [sc_k241_r7100584_c_k31l75.w0.json](sc_k241_r7100584_c_k31l75.w0.json) · result
- [sc_k241_r7100584_c_k4l90.w1.json](sc_k241_r7100584_c_k4l90.w1.json) · result
- [sc_k241_r7100585_c_f4c85.w0.json](sc_k241_r7100585_c_f4c85.w0.json) · result
- [sc_k241_r7100585_c_k31l75.w1.json](sc_k241_r7100585_c_k31l75.w1.json) · result
- [sc_k241_r7100585_c_k4l90.w0.json](sc_k241_r7100585_c_k4l90.w0.json) · result
- [sc_k241_r7100586_c_f4c85.w1.json](sc_k241_r7100586_c_f4c85.w1.json) · result
- [sc_k241_r7100586_c_k31l75.w0.json](sc_k241_r7100586_c_k31l75.w0.json) · result
- [sc_k241_r7100586_c_k4l90.w1.json](sc_k241_r7100586_c_k4l90.w1.json) · result
- [sc_k241_r7100587_c_f4c85.w0.json](sc_k241_r7100587_c_f4c85.w0.json) · result
- [sc_k241_r7100587_c_k31l75.w1.json](sc_k241_r7100587_c_k31l75.w1.json) · result
- [sc_k241_r7100587_c_k4l90.w0.json](sc_k241_r7100587_c_k4l90.w0.json) · result
- [sc_k241_r7100588_c_f4c85.w1.json](sc_k241_r7100588_c_f4c85.w1.json) · result
- [sc_k241_r7100588_c_k31l75.w0.json](sc_k241_r7100588_c_k31l75.w0.json) · result
- [sc_k241_r7100588_c_k4l90.w1.json](sc_k241_r7100588_c_k4l90.w1.json) · result
- [sc_k241_r7100589_c_f4c85.w0.json](sc_k241_r7100589_c_f4c85.w0.json) · result
- [sc_k241_r7100589_c_k31l75.w1.json](sc_k241_r7100589_c_k31l75.w1.json) · result
- [sc_k241_r7100589_c_k4l90.w0.json](sc_k241_r7100589_c_k4l90.w0.json) · result
- [sc_k241_r7100590_c_f4c85.w1.json](sc_k241_r7100590_c_f4c85.w1.json) · result
- [sc_k241_r7100590_c_k31l75.w0.json](sc_k241_r7100590_c_k31l75.w0.json) · result
- [sc_k241_r7100590_c_k4l90.w1.json](sc_k241_r7100590_c_k4l90.w1.json) · result
- [sc_k241_r7100591_c_f4c85.w0.json](sc_k241_r7100591_c_f4c85.w0.json) · result
- [sc_k241_r7100591_c_k31l75.w1.json](sc_k241_r7100591_c_k31l75.w1.json) · result
- [sc_k241_r7100591_c_k4l90.w0.json](sc_k241_r7100591_c_k4l90.w0.json) · result
- [sc_k241_r7100584_c_k4b90.json](sc_k241_r7100584_c_k4b90.json) · result
- [sc_k241_r7100585_c_k4b90.json](sc_k241_r7100585_c_k4b90.json) · result
- [sc_k241_r7100586_c_k4b90.json](sc_k241_r7100586_c_k4b90.json) · result
- [sc_k241_r7100587_c_k4b90.json](sc_k241_r7100587_c_k4b90.json) · result
- [sc_k241_r7100588_c_k4b90.json](sc_k241_r7100588_c_k4b90.json) · result
- [sc_k241_r7100589_c_k4b90.json](sc_k241_r7100589_c_k4b90.json) · result
- [sc_k241_r7100590_c_k4b90.json](sc_k241_r7100590_c_k4b90.json) · result
- [sc_k241_r7100591_c_k4b90.json](sc_k241_r7100591_c_k4b90.json) · result
- [guard_v3_w0_20261013T153902Z.json](guard_v3_w0_20261013T153902Z.json) · result
- [guard_v3_w1_20261013T153922Z.json](guard_v3_w1_20261013T153922Z.json) · result
- [sc_k240_r7100584_cbase.json](sc_k240_r7100584_cbase.json) · result
- [sc_k241_r7100584_c_f4c85.json](sc_k241_r7100584_c_f4c85.json) · result
- [sc_k241_r7100584_c_k31l75.json](sc_k241_r7100584_c_k31l75.json) · result
- [sc_k241_r7100584_c_k4l90.json](sc_k241_r7100584_c_k4l90.json) · result
- [sc_k241_r7100585_c_f4c85.json](sc_k241_r7100585_c_f4c85.json) · result
- [sc_k241_r7100585_c_k31l75.json](sc_k241_r7100585_c_k31l75.json) · result
- [sc_k241_r7100585_c_k4l90.json](sc_k241_r7100585_c_k4l90.json) · result
- [sc_k241_r7100586_c_f4c85.json](sc_k241_r7100586_c_f4c85.json) · result
- [sc_k241_r7100586_c_k31l75.json](sc_k241_r7100586_c_k31l75.json) · result
- [sc_k241_r7100586_c_k4l90.json](sc_k241_r7100586_c_k4l90.json) · result
- [sc_k241_r7100587_c_f4c85.json](sc_k241_r7100587_c_f4c85.json) · result
- [sc_k241_r7100587_c_k31l75.json](sc_k241_r7100587_c_k31l75.json) · result
- [sc_k241_r7100587_c_k4l90.json](sc_k241_r7100587_c_k4l90.json) · result
- [sc_k241_r7100588_c_f4c85.json](sc_k241_r7100588_c_f4c85.json) · result
- [sc_k241_r7100588_c_k31l75.json](sc_k241_r7100588_c_k31l75.json) · result
- [sc_k241_r7100588_c_k4l90.json](sc_k241_r7100588_c_k4l90.json) · result
- [sc_k241_r7100589_c_f4c85.json](sc_k241_r7100589_c_f4c85.json) · result
- [sc_k241_r7100589_c_k31l75.json](sc_k241_r7100589_c_k31l75.json) · result
- [sc_k241_r7100589_c_k4l90.json](sc_k241_r7100589_c_k4l90.json) · result
- [sc_k241_r7100590_c_f4c85.json](sc_k241_r7100590_c_f4c85.json) · result
- [sc_k241_r7100590_c_k31l75.json](sc_k241_r7100590_c_k31l75.json) · result
- [sc_k241_r7100590_c_k4l90.json](sc_k241_r7100590_c_k4l90.json) · result
- [sc_k241_r7100591_c_f4c85.json](sc_k241_r7100591_c_f4c85.json) · result
- [sc_k241_r7100591_c_k31l75.json](sc_k241_r7100591_c_k31l75.json) · result
- [sc_k241_r7100591_c_k4l90.json](sc_k241_r7100591_c_k4l90.json) · result
- [worker_w0-5290757f.log](worker_w0-5290757f.log) · log
- [worker_w1.log](worker_w1.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [t144_table.out](t144_table.out) · result
- [t143_k4_residual.out](t143_k4_residual.out) · result

</details>

## Related work

- ← enables: [Hub quartic ages 3-4 extended to layers 8-15: -0.513% mean-log MSE on 24 networks](../../../records/research/2026-10-13-hub-quartic-ages-3-4-extended-layers-8-15/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-additional-fourth-order-cohort-ages-overlapping-strengths-rejected-after) · [Copy provenance](../../../PROVENANCE.json)

