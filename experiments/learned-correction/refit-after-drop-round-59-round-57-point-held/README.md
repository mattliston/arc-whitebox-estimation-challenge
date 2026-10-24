# Refit-after-drop round 59 from the round-57 point: held-out adjusted -0.008%, rejected

2026-10-14T14:43:37Z · experiment · rejected · measured

The 59th refit-after-drop round, again from the S57 frozen point with the four lean arms, froze a weak point whose sealed check was flat, so it was not accepted and the chain counted its first dry round (the worker count rule drops the line to two workers after two dry rounds). Round S60 was started from the S59 point.

Recorded measurement:

    frozen point: LOO -0.008% (noise-corrected -0.013%), model -0.022%, ridge 0.5, weights arrp 9.89, fagg -2.95; CHECK on 7103400-7103431: frozen vs start -0.008% (se 0.022, better 17/32) -> NOT ACCEPTED; sizing 320 evals in 144 min; dry rounds in a row = 1

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
<summary>Additional programs and dependencies (45)</summary>

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
- [code/opus_reexam_20261001_v1/radlab/aggregate_spec_v1.py](code/opus_reexam_20261001_v1/radlab/aggregate_spec_v1.py)
- [code/opus_reexam_20261001_v1/radlab/ptload.py](code/opus_reexam_20261001_v1/radlab/ptload.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/S59/make_s59_v16.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/S59/make_s59_v16.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/absorb_v1.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/absorb_v1.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/absorb_v3.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/absorb_v3.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/zero_arms_v1.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/zero_arms_v1.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_v7.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_v7.py)
- [code/opus_reexam_20261001_v1/trainer/export.py](code/opus_reexam_20261001_v1/trainer/export.py)
- [code/opus_reexam_20261001_v1/trainer/export_dials_s46_v12.py](code/opus_reexam_20261001_v1/trainer/export_dials_s46_v12.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m_v7.py](code/opus_reexam_20261001_v1/trainer/otr_m_v7.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl_v2.py](code/opus_reexam_20261001_v1/trainer/otr_pl_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py](code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py)
- [code/opus_reexam_20261001_v1/trainer/worker_generated_v16.py](code/opus_reexam_20261001_v1/trainer/worker_generated_v16.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2849](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2849)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2854](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2854)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2865](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2865)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2875](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2875)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2876](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2876)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2877](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2877)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2878](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2878)

<details>
<summary>All result files and records (3)</summary>

- [FROZEN_POINT_v1.json](FROZEN_POINT_v1.json) · result
- [CHECK_VERDICT_v1.json](CHECK_VERDICT_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-refit-after-drop-round-59-round-57-point-held) · [Copy provenance](../../../PROVENANCE.json)

