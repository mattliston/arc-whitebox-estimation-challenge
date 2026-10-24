# Additional layer-15 Edgeworth and fourth-order terms rejected: error increases or gains miss acceptance

2026-10-13T08:37Z · experiment · rejected · measured

The mean formula's next terms at layer 15 (k3^2 He4 and k3 k4 He5) make the final error worse on top of final-layer hub quartic ages 3-4 at strength 0.5 (k4lf34h) because the learned head already holds them; F4C age 3 at the age-2 strengths costs +0.47%; the F4 shift's nonlinear remainder is 1-3% of the linear term; the learned-off so2m / so2t screen stays below the 1% bar.

Recorded measurement:

    e2 +0.434% (se 0.315), e34 +0.347%, both +0.595% (2.4 se); e2 at -0.5 +0.197%; F4C age 3 +0.466% (se 0.546); N vs L 0.9-1.3% first chaos, 2.1-3.1% K21; so2m -0.677% (se 0.766), so2t -0.134%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/physicist/f4b_t138.py](code/opus_reexam_20261001_v1/physicist/f4b_t138.py)
- [code/opus_reexam_20261001_v1/physicist/t134_table.py](code/opus_reexam_20261001_v1/physicist/t134_table.py)
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
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc6.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc6.py)
- [code/opus_reexam_20261001_v1/scout_card/worker_sc10.py](code/opus_reexam_20261001_v1/scout_card/worker_sc10.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L1974](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1974)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1976](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1976)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1988](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1988)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1989](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1989)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1994](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1994)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1995](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1995)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1996](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1996)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2005](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2005)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2021](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2021)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2029](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2029)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2030](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2030)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1902](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1902)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1903](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1903)

<details>
<summary>All result files and records (56)</summary>

- [t138_f4diag.jsonl](t138_f4diag.jsonl) · result
- [sc_k190_r7100584_s51k4lf34h.w0.json](sc_k190_r7100584_s51k4lf34h.w0.json) · result
- [sc_k191_r7100584_s51e2.w1.json](sc_k191_r7100584_s51e2.w1.json) · result
- [sc_k191_r7100584_s51e2e34.w0.json](sc_k191_r7100584_s51e2e34.w0.json) · result
- [sc_k191_r7100584_s51e34.w1.json](sc_k191_r7100584_s51e34.w1.json) · result
- [sc_k191_r7100585_s51e2.w0.json](sc_k191_r7100585_s51e2.w0.json) · result
- [sc_k191_r7100585_s51e2e34.w1.json](sc_k191_r7100585_s51e2e34.w1.json) · result
- [sc_k191_r7100585_s51e34.w0.json](sc_k191_r7100585_s51e34.w0.json) · result
- [sc_k191_r7100586_s51e2.w1.json](sc_k191_r7100586_s51e2.w1.json) · result
- [sc_k191_r7100586_s51e2e34.w0.json](sc_k191_r7100586_s51e2e34.w0.json) · result
- [sc_k191_r7100586_s51e34.w1.json](sc_k191_r7100586_s51e34.w1.json) · result
- [sc_k191_r7100587_s51e2.w0.json](sc_k191_r7100587_s51e2.w0.json) · result
- [sc_k191_r7100587_s51e2e34.w1.json](sc_k191_r7100587_s51e2e34.w1.json) · result
- [sc_k191_r7100587_s51e34.w0.json](sc_k191_r7100587_s51e34.w0.json) · result
- [sc_k191_r7100588_s51e2.w1.json](sc_k191_r7100588_s51e2.w1.json) · result
- [sc_k191_r7100588_s51e2e34.w0.json](sc_k191_r7100588_s51e2e34.w0.json) · result
- [sc_k191_r7100588_s51e34.w1.json](sc_k191_r7100588_s51e34.w1.json) · result
- [sc_k191_r7100589_s51e2.w0.json](sc_k191_r7100589_s51e2.w0.json) · result
- [sc_k191_r7100589_s51e2e34.w1.json](sc_k191_r7100589_s51e2e34.w1.json) · result
- [sc_k191_r7100589_s51e34.w0.json](sc_k191_r7100589_s51e34.w0.json) · result
- [sc_k191_r7100590_s51e2.w1.json](sc_k191_r7100590_s51e2.w1.json) · result
- [sc_k191_r7100590_s51e2e34.w0.json](sc_k191_r7100590_s51e2e34.w0.json) · result
- [sc_k191_r7100590_s51e34.w1.json](sc_k191_r7100590_s51e34.w1.json) · result
- [sc_k191_r7100591_s51e2.w0.json](sc_k191_r7100591_s51e2.w0.json) · result
- [sc_k191_r7100591_s51e2e34.w1.json](sc_k191_r7100591_s51e2e34.w1.json) · result
- [sc_k191_r7100591_s51e34.w0.json](sc_k191_r7100591_s51e34.w0.json) · result
- [guard_v3_w0_20261013T070502Z.json](guard_v3_w0_20261013T070502Z.json) · result
- [guard_v3_w1_20261013T070523Z.json](guard_v3_w1_20261013T070523Z.json) · result
- [sc_k190_r7100584_s51k4lf34h.json](sc_k190_r7100584_s51k4lf34h.json) · result
- [sc_k191_r7100584_s51e2.json](sc_k191_r7100584_s51e2.json) · result
- [sc_k191_r7100584_s51e2e34.json](sc_k191_r7100584_s51e2e34.json) · result
- [sc_k191_r7100584_s51e34.json](sc_k191_r7100584_s51e34.json) · result
- [sc_k191_r7100585_s51e2.json](sc_k191_r7100585_s51e2.json) · result
- [sc_k191_r7100585_s51e2e34.json](sc_k191_r7100585_s51e2e34.json) · result
- [sc_k191_r7100585_s51e34.json](sc_k191_r7100585_s51e34.json) · result
- [sc_k191_r7100586_s51e2.json](sc_k191_r7100586_s51e2.json) · result
- [sc_k191_r7100586_s51e2e34.json](sc_k191_r7100586_s51e2e34.json) · result
- [sc_k191_r7100586_s51e34.json](sc_k191_r7100586_s51e34.json) · result
- [sc_k191_r7100587_s51e2.json](sc_k191_r7100587_s51e2.json) · result
- [sc_k191_r7100587_s51e2e34.json](sc_k191_r7100587_s51e2e34.json) · result
- [sc_k191_r7100587_s51e34.json](sc_k191_r7100587_s51e34.json) · result
- [sc_k191_r7100588_s51e2.json](sc_k191_r7100588_s51e2.json) · result
- [sc_k191_r7100588_s51e2e34.json](sc_k191_r7100588_s51e2e34.json) · result
- [sc_k191_r7100588_s51e34.json](sc_k191_r7100588_s51e34.json) · result
- [sc_k191_r7100589_s51e2.json](sc_k191_r7100589_s51e2.json) · result
- [sc_k191_r7100589_s51e2e34.json](sc_k191_r7100589_s51e2e34.json) · result
- [sc_k191_r7100589_s51e34.json](sc_k191_r7100589_s51e34.json) · result
- [sc_k191_r7100590_s51e2.json](sc_k191_r7100590_s51e2.json) · result
- [sc_k191_r7100590_s51e2e34.json](sc_k191_r7100590_s51e2e34.json) · result
- [sc_k191_r7100590_s51e34.json](sc_k191_r7100590_s51e34.json) · result
- [sc_k191_r7100591_s51e2.json](sc_k191_r7100591_s51e2.json) · result
- [sc_k191_r7100591_s51e2e34.json](sc_k191_r7100591_s51e2e34.json) · result
- [sc_k191_r7100591_s51e34.json](sc_k191_r7100591_s51e34.json) · result
- [worker_w0.log](worker_w0.log) · log
- [worker_w1.log](worker_w1.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Final-layer hub quartic ages 3-4 at strength 0.5 adopted: -0.247% mean-log MSE on 24 networks](../../../records/research/2026-10-13-final-layer-hub-quartic-ages-3-4-strength-0/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-additional-layer-15-edgeworth-fourth-order-terms-rejected-error) · [Copy provenance](../../../PROVENANCE.json)

