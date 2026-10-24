# Inner-quartic re-screens reject per-unit covariance-squared kurtosis and exact first-order fourth-chaos terms

2026-10-13T11:26Z · experiment · rejected · measured

Two closures from 10-01..10-10 were re-screened on top of C (cohort-age local fourth-order term (K4L) ages 3-4 at 0.5 at layers 8-15 plus the final-layer term) at the candidate generation 51 (S51) + transport-gate point on 8 fresh networks: the per-unit kurtosis correction k4 += (6 b / n) (Cpre^2)_ii at layers 8-14 with b = 0.0625 (E33) is much worse, and the F4C plane's exact first-order term (x1a1 = x1a2, off since 10-02) at 0.25 at layers 10-14 is within noise against C, so neither reached the -0.15% screen bar.

Recorded measurement:

    c_e33 vs C: pooled +2.147%, mean log +2.102% (se 1.215, 1.7 SE), 3 of 8 lower, all-layer +0.820%; vs k4lf34h +1.633%; c_x1a vs C: pooled -0.105%, mean log -0.072% (se 0.120, 0.6 SE), 5 of 8 lower; vs k4lf34h -0.541% (se 0.280)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

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
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (47)</summary>

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
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/scout_card/chain_k22_after_t142_v1.sh](code/opus_reexam_20261001_v1/scout_card/chain_k22_after_t142_v1.sh)
- [code/opus_reexam_20261001_v1/scout_card/k1_chain_v13.sh](code/opus_reexam_20261001_v1/scout_card/k1_chain_v13.sh)
- [code/opus_reexam_20261001_v1/scout_card/k_line_v1.py](code/opus_reexam_20261001_v1/scout_card/k_line_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/make_k18_v7.py](code/opus_reexam_20261001_v1/scout_card/make_k18_v7.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_cut_runtime_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_cut_runtime_sc4.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc7.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc7.py)
- [code/opus_reexam_20261001_v1/scout_card/worker_sc11.py](code/opus_reexam_20261001_v1/scout_card/worker_sc11.py)
- [code/opus_reexam_20261001_v1/scout_card/worker_sc8.py](code/opus_reexam_20261001_v1/scout_card/worker_sc8.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L2073](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2073)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2074](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2074)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2059](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2059)

<details>
<summary>All result files and records (41)</summary>

- [STOP](STOP) · result
- [chain.out](chain.out) · result
- [sc_k220_r7100584_cbase.w0.json](sc_k220_r7100584_cbase.w0.json) · result
- [sc_k221_r7100584_c_e33.w1.json](sc_k221_r7100584_c_e33.w1.json) · result
- [sc_k221_r7100584_c_x1a.w0.json](sc_k221_r7100584_c_x1a.w0.json) · result
- [sc_k221_r7100585_c_e33.w1.json](sc_k221_r7100585_c_e33.w1.json) · result
- [sc_k221_r7100585_c_x1a.w0.json](sc_k221_r7100585_c_x1a.w0.json) · result
- [sc_k221_r7100586_c_e33.w1.json](sc_k221_r7100586_c_e33.w1.json) · result
- [sc_k221_r7100586_c_x1a.w0.json](sc_k221_r7100586_c_x1a.w0.json) · result
- [sc_k221_r7100587_c_e33.w1.json](sc_k221_r7100587_c_e33.w1.json) · result
- [sc_k221_r7100587_c_x1a.w0.json](sc_k221_r7100587_c_x1a.w0.json) · result
- [sc_k221_r7100588_c_e33.w1.json](sc_k221_r7100588_c_e33.w1.json) · result
- [sc_k221_r7100588_c_x1a.w0.json](sc_k221_r7100588_c_x1a.w0.json) · result
- [sc_k221_r7100589_c_e33.w1.json](sc_k221_r7100589_c_e33.w1.json) · result
- [sc_k221_r7100589_c_x1a.w0.json](sc_k221_r7100589_c_x1a.w0.json) · result
- [sc_k221_r7100590_c_e33.w1.json](sc_k221_r7100590_c_e33.w1.json) · result
- [sc_k221_r7100590_c_x1a.w0.json](sc_k221_r7100590_c_x1a.w0.json) · result
- [sc_k221_r7100591_c_e33.w1.json](sc_k221_r7100591_c_e33.w1.json) · result
- [sc_k221_r7100591_c_x1a.w0.json](sc_k221_r7100591_c_x1a.w0.json) · result
- [guard_v3_w0_20261013T110416Z.json](guard_v3_w0_20261013T110416Z.json) · result
- [guard_v3_w1_20261013T110437Z.json](guard_v3_w1_20261013T110437Z.json) · result
- [sc_k220_r7100584_cbase.json](sc_k220_r7100584_cbase.json) · result
- [sc_k221_r7100584_c_e33.json](sc_k221_r7100584_c_e33.json) · result
- [sc_k221_r7100584_c_x1a.json](sc_k221_r7100584_c_x1a.json) · result
- [sc_k221_r7100585_c_e33.json](sc_k221_r7100585_c_e33.json) · result
- [sc_k221_r7100585_c_x1a.json](sc_k221_r7100585_c_x1a.json) · result
- [sc_k221_r7100586_c_e33.json](sc_k221_r7100586_c_e33.json) · result
- [sc_k221_r7100586_c_x1a.json](sc_k221_r7100586_c_x1a.json) · result
- [sc_k221_r7100587_c_e33.json](sc_k221_r7100587_c_e33.json) · result
- [sc_k221_r7100587_c_x1a.json](sc_k221_r7100587_c_x1a.json) · result
- [sc_k221_r7100588_c_e33.json](sc_k221_r7100588_c_e33.json) · result
- [sc_k221_r7100588_c_x1a.json](sc_k221_r7100588_c_x1a.json) · result
- [sc_k221_r7100589_c_e33.json](sc_k221_r7100589_c_e33.json) · result
- [sc_k221_r7100589_c_x1a.json](sc_k221_r7100589_c_x1a.json) · result
- [sc_k221_r7100590_c_e33.json](sc_k221_r7100590_c_e33.json) · result
- [sc_k221_r7100590_c_x1a.json](sc_k221_r7100590_c_x1a.json) · result
- [sc_k221_r7100591_c_e33.json](sc_k221_r7100591_c_e33.json) · result
- [sc_k221_r7100591_c_x1a.json](sc_k221_r7100591_c_x1a.json) · result
- [worker_w0.log](worker_w0.log) · log
- [worker_w1.log](worker_w1.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-inner-quartic-re-screens-reject-per-unit-covariance-squared) · [Copy provenance](../../../PROVENANCE.json)

