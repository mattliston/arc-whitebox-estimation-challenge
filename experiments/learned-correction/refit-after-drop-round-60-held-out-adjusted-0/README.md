# Refit-after-drop round 60: held-out adjusted -0.003%; two unproductive rounds end the line

2026-10-14T20:51:31Z · experiment · inconclusive · measured

Round S60 froze a heavily regularised four-weight point whose sealed check passed the acceptance rule by a negligible margin; with S59 it made two dry rounds, which by the pre-declared rule capped the refit line at two workers and freed the CPU for error-side work. The lineage S57/S59/S60 estimates multiply to about -0.044% and the S60 increment alone is -0.0035%, so no port was promoted.

Recorded measurement:

    frozen point: LOO -0.001% (noise-corrected -0.001%), model -0.004%, ridge 10.0, weights arrp 1.07, dkc_7_10 0.21, dx2a1_7_7 -0.29, dk4la3_4_7 0.66; CHECK on 7103784-7103815: frozen vs start -0.003% (se 0.002, better 19/32) -> ACCEPTED; sizing 320 evals in 200 min; STOP/SLOW after S60: WORKER_CAP = 2; cached lineage S57/S59/S60 ~ -0.04438%, S60 increment -0.003491%

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
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/S60/make_s60_v16.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/S60/make_s60_v16.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L2911](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2911)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3037](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3037)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3072](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3072)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3073](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3073)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3078](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3078)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3079](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3079)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3227](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3227)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3230](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3230)

<details>
<summary>All result files and records (4)</summary>

- [FROZEN_POINT_v1.json](FROZEN_POINT_v1.json) · result
- [CHECK_VERDICT_v1.json](CHECK_VERDICT_v1.json) · result
- [CACHED_GAIN_AND_CURVATURE_LIMIT_v1.json](CACHED_GAIN_AND_CURVATURE_LIMIT_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-refit-after-drop-round-60-held-out-adjusted-0) · [Copy provenance](../../../PROVENANCE.json)

