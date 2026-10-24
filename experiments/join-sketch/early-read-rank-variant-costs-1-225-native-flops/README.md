# Early-read rank variant costs +1.225% native FLOPs versus the copy baseline; screen remains pending

2026-10-13T23:29Z · experiment · inconclusive · measured

A cheaper form of the rejected L10 rank-384 restoration was registered and priced: keep the full 384-column state, common 256 and late reconstruction but read only 320 columns at layers 10-13, with no refit or rank search. Its six-case native run (parent / off / on at networks 7100136 and 7100137) froze the worst-case metered FLOPs penalty at +1.225% in log terms against the compute-copy baseline, about 0.334% cheaper than the full rank 384 at the worst seed; the fixed fresh-16 accuracy screen was granted but had not run inside this slice, so no net verdict exists.

Recorded measurement:

    frozen actual MAX bill penalty +1.225440016 percent log vs the current copy baseline; roughly 0.334 percent log cheaper than full 384 at the worst seed; all 16 OFF identities exact; one scheduling defect (outer nice 10 compounded the guard's nice 10) corrected before the remaining four cases

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
<summary>Additional programs and dependencies (67)</summary>

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
- [code/opus_reexam_20261001_v1/engineer/a62_modules/parity_verdict_v5.py](code/opus_reexam_20261001_v1/engineer/a62_modules/parity_verdict_v5.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/candidate_early_vread_noop_UNVALIDATED_astra_v1_v3.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/candidate_early_vread_noop_UNVALIDATED_astra_v1_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/candidate_early_vread_on_UNVALIDATED_astra_v1_v3.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/candidate_early_vread_on_UNVALIDATED_astra_v1_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/candidate_parent_full384_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/candidate_parent_full384_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_control/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_control/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_control/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_control/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_control/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_control/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_control/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_control/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_control/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_control/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_early_vread/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_early_vread/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_early_vread/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_early_vread/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_early_vread/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_early_vread/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_early_vread/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_early_vread/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_early_vread/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_early_vread/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_rank384_zero/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_rank384_zero/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_rank384_zero/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_rank384_zero/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_rank384_zero/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_rank384_zero/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_rank384_zero/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_rank384_zero/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_rank384_zero/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/cutsim_rank384_zero/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/fragment_early_vread_noop_UNVALIDATED_astra_v1_v3.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/fragment_early_vread_noop_UNVALIDATED_astra_v1_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/fragment_early_vread_on_UNVALIDATED_astra_v1_v3.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/fragment_early_vread_on_UNVALIDATED_astra_v1_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/runtime_early_vread_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/runtime_early_vread_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/worker_early_vread_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/worker_early_vread_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/freeze_native_price_retirement_astra_v2.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/freeze_native_price_retirement_astra_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/launch_early_vread_astra_v3.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/launch_early_vread_astra_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/prepare_early_vread_astra_v3.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/prepare_early_vread_astra_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/run_native_price_early_vread_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/run_native_price_early_vread_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/run_native_price_early_vread_astra_v2.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/run_native_price_early_vread_astra_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/table_early_vread_astra_v3.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/table_early_vread_astra_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/fragment_early_vread_UNVALIDATED_astra_v1_v3.py](code/opus_reexam_20261001_v1/scout_card/fragment_early_vread_UNVALIDATED_astra_v1_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py)
- [code/opus_reexam_20261001_v1/scout_card/run_rank_native_price_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/run_rank_native_price_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/astra_compute_v1_arm/candidate.py](code/opus_reexam_20261001_v1/scout_native/astra_compute_v1_arm/candidate.py)
- [code/opus_reexam_20261001_v1/scout_native/run_cert_v2.py](code/opus_reexam_20261001_v1/scout_native/run_cert_v2.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L2333](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2333)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2409](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2409)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2396](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2396)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2398](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2398)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-early-read-rank-variant-costs-1-225-native-flops) · [Copy provenance](../../../PROVENANCE.json)

