# Raw (2,1) covariance target at layers 1-14: fresh-32 MSE +7.76% at bill +0.11%

2026-10-14T05:39:57Z · experiment · rejected · measured

Retargeting the raw R21 term at the actual outgoing covariance (fourteen target contexts per network) passed its exactness and price gates but raised the final MSE by almost 8% on the fresh screen.

Recorded measurement:

    native MAX log bill 0.0011447374104286345; fresh 32 (16 pairs): mse_log 7.7566563438519545%, bill_log 0.11447374104286345%, net_log 7.871130084894817% (SE 1.0407579933833635%)

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
<summary>Additional programs and dependencies (49)</summary>

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
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_reconstruction_row_allocation_v1/science_held_v1/finite_arow_science_UNVALIDATED_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_reconstruction_row_allocation_v1/science_held_v1/finite_arow_science_UNVALIDATED_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_reconstruction_row_allocation_v1/science_held_v1/finite_arow_smoke_UNVALIDATED_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_reconstruction_row_allocation_v1/science_held_v1/finite_arow_smoke_UNVALIDATED_v2.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/build_native_source_astra_v2.py](code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/build_native_source_astra_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/build_raw21_science_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/build_raw21_science_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/finite_raw21_science_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/finite_raw21_science_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/finite_raw21_science_UNVALIDATED_astra_v2.py](code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/finite_raw21_science_UNVALIDATED_astra_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/finite_raw21_smoke_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/finite_raw21_smoke_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/fragment_raw21_UNVALIDATED_astra_v2.py](code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/fragment_raw21_UNVALIDATED_astra_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/isolated_raw21_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/isolated_raw21_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/native_OFF_UNVALIDATED_astra_v2.py](code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/native_OFF_UNVALIDATED_astra_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/native_ON_UNVALIDATED_astra_v2.py](code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/native_ON_UNVALIDATED_astra_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/torch_raw21_target_UNVALIDATED_astra_v3.py](code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/torch_raw21_target_UNVALIDATED_astra_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/worker_raw21_target_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/k21_raw_target_held_astra_v1/worker_raw21_target_UNVALIDATED_astra_v1.py)
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

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2656](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2656)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2668](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2668)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2680](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2680)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2719](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2719)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-raw-2-1-covariance-target-layers-1-14-fresh) · [Copy provenance](../../../PROVENANCE.json)

