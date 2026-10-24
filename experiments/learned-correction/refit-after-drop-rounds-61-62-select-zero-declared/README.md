# Refit-after-drop rounds 61 and 62 select zero by the declared leave-one-out criterion

2026-10-15T04:35:28Z · experiment · rejected · measured

With the feasible head mean (S61) and then the output projection (S62) absorbed, the lean refit rounds found no gradient arm worth a step: both frozen points were null (empty weights) by the pre-declared leave-one-out rule, so no check ran. The lead's two auxiliary helper workers for S61 were withdrawn at the owner's compute priority; round S63 started from the S60 point on two CPU workers.

Recorded measurement:

    S61: LOO +0.000% (noise-corrected +0.001%), model +0.000%, ridge 10.0, null True, weights {}; sizing 320 evals in 169 min; S62: LOO +0.002% (noise-corrected +0.003%), model +0.000%, null True; 256 evals in 146 min; S61 helpers: 36 results preserved, 2 interrupted requests returned

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
<summary>Additional programs and dependencies (52)</summary>

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
- [code/opus_reexam_20261001_v1/lead/runs/astra_s61_finite_aux_v1/finite_s61_aux_controller_HELD_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_s61_finite_aux_v1/finite_s61_aux_controller_HELD_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_s61_finite_aux_v1/finite_s61_aux_worker_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_s61_finite_aux_v1/finite_s61_aux_worker_HELD_v1.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/radlab/aggregate_spec_v1.py](code/opus_reexam_20261001_v1/radlab/aggregate_spec_v1.py)
- [code/opus_reexam_20261001_v1/radlab/ptload.py](code/opus_reexam_20261001_v1/radlab/ptload.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/scout_native/certify_astra_bank_copy_v2.py](code/opus_reexam_20261001_v1/scout_native/certify_astra_bank_copy_v2.py)
- [code/opus_reexam_20261001_v1/scout_native/run_cert_astra_bank_copy_v2.py](code/opus_reexam_20261001_v1/scout_native/run_cert_astra_bank_copy_v2.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/S61/make_s61_v16.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/S61/make_s61_v16.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/S62/make_s62_v16.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/S62/make_s62_v16.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/absorb_v1.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/absorb_v1.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/absorb_v3.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/absorb_v3.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/zero_arms_v1.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/zero_arms_v1.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_v7.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_v7.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m_v7.py](code/opus_reexam_20261001_v1/trainer/otr_m_v7.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m_v8.py](code/opus_reexam_20261001_v1/trainer/otr_m_v8.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl_v2.py](code/opus_reexam_20261001_v1/trainer/otr_pl_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py](code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py)
- [code/opus_reexam_20261001_v1/trainer/worker_generated_v16.py](code/opus_reexam_20261001_v1/trainer/worker_generated_v16.py)
- [code/opus_reexam_20261001_v1/trainer/worker_generated_v17.py](code/opus_reexam_20261001_v1/trainer/worker_generated_v17.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3154](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3154)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3225](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3225)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3228](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3228)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3230](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3230)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3231](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3231)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3233](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3233)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3234](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3234)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3236](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3236)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3318](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3318)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3320](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3320)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3321](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3321)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3322](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3322)

<details>
<summary>All result files and records (4)</summary>

- [FROZEN_POINT_v1.json](../../../records/research/2026-10-15-feasible-mean-rank-two-output-projection-absorbed-into-refit/FROZEN_POINT_v1.json) · result
- [FROZEN_POINT_v1-c0e34da7.json](../../../records/research/2026-10-15-feasible-mean-rank-two-output-projection-absorbed-into-refit/FROZEN_POINT_v1-c0e34da7.json) · result
- [OWNER_WITHDRAWAL_RETIREMENT_v1.json](OWNER_WITHDRAWAL_RETIREMENT_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-refit-after-drop-rounds-61-62-select-zero-declared) · [Copy provenance](../../../PROVENANCE.json)

