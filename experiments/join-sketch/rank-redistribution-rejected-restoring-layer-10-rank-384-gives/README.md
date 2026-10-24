# Rank redistribution rejected: restoring layer-10 rank 384 gives +0.148% net out-of-selection change

2026-10-13T23:04Z · experiment · rejected · measured

Astra's blinded capacity continuations on top of C: restoring rank 384 at L10 with a byte-identical cache gains MSE on the out-of-selection 16 but the frozen native cost makes the net positive; funding L10 from L7 (384 -> 320) survives the fresh 16 at -0.084% net but fails the out-of-selection (OOS) 16 at +0.209%.

Recorded measurement:

    UB384: OOS mean log MSE -1.417%, net +0.148%; combined 24 MSE -1.532%, net +0.033%; fund64 fresh16 MSE +1.100%, bill -1.184%, net -0.084%; OOS16 MSE +1.393%, net +0.209%; all32 net +0.062%

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
<summary>Additional programs and dependencies (109)</summary>

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
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fastgate_post_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fastgate_post_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_xsg_any_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_xsg_any_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py](code/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s54k1tw_x/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s54k1tw_x/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/quick.py](code/opus_reexam_20261001_v1/engineer/quick.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_push_native_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/runtime_weightfit_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/runtime_weightfit_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/worker_weightfit_finite_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/worker_weightfit_finite_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_rank_redistribution_v1/launch_rank_redistribution_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_rank_redistribution_v1/launch_rank_redistribution_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_rank_redistribution_v1/prepare_rank_redistribution_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_rank_redistribution_v1/prepare_rank_redistribution_v3.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_rank_redistribution_v1/prioritize_shared_controls_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_rank_redistribution_v1/prioritize_shared_controls_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_rank_redistribution_v1/table_rank_redistribution_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_rank_redistribution_v1/table_rank_redistribution_v3.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_held_native_astra_v1/candidate_rank384_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/capacity_held_native_astra_v1/candidate_rank384_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_held_ub384_astra_v1_v2/candidate_ub384_noop_UNVALIDATED_astra_v1_v2.py](code/opus_reexam_20261001_v1/scout_card/capacity_held_ub384_astra_v1_v2/candidate_ub384_noop_UNVALIDATED_astra_v1_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_held_ub384_astra_v1_v2/candidate_ub384_on_UNVALIDATED_astra_v1_v2.py](code/opus_reexam_20261001_v1/scout_card/capacity_held_ub384_astra_v1_v2/candidate_ub384_on_UNVALIDATED_astra_v1_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc10_astra_v1/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc10_astra_v1/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc10_astra_v1/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc10_astra_v1/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc10_astra_v1/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc10_astra_v1/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc10_astra_v1/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc10_astra_v1/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc10_astra_v1/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc10_astra_v1/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/fragment_capacity_rank384_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/fragment_capacity_rank384_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/fragment_capacity_ub384_noop_UNVALIDATED_astra_v1_v2.py](code/opus_reexam_20261001_v1/scout_card/fragment_capacity_ub384_noop_UNVALIDATED_astra_v1_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/fragment_capacity_ub384_on_UNVALIDATED_astra_v1_v2.py](code/opus_reexam_20261001_v1/scout_card/fragment_capacity_ub384_on_UNVALIDATED_astra_v1_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/candidate_fund64_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/candidate_fund64_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/candidate_transfer64_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/candidate_transfer64_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/candidate_transfer64_off_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/candidate_transfer64_off_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/candidate_zero_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/candidate_zero_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/fragment_fund64_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/fragment_fund64_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/fragment_transfer64_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/fragment_transfer64_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/fragment_transfer64_off_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/fragment_transfer64_off_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/fragment_zero_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_native_price_held_astra_v1/fragment_zero_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_fund/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_fund/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_fund/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_fund/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_fund/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_fund/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_fund/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_fund/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_fund/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_fund/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_joint/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_joint/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_joint/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_joint/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_joint/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_joint/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_joint/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_joint/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_joint/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_joint/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_zero/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_zero/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_zero/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_zero/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_zero/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_zero/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_zero/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_zero/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_zero/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/cutsim_zero/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/fragment_rank_fund_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/fragment_rank_fund_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/fragment_rank_joint_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/fragment_rank_joint_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/fragment_rank_zero_UNVALIDATED_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/fragment_rank_zero_UNVALIDATED_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/runtime_rank_redistribution_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/runtime_rank_redistribution_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/worker_rank_redistribution_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/rank_redistribution_held_astra_v1/worker_rank_redistribution_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/astra_compute_v1_arm/candidate.py](code/opus_reexam_20261001_v1/scout_native/astra_compute_v1_arm/candidate.py)
- [code/opus_reexam_20261001_v1/scout_native/chain_round_stack_v4.sh](code/opus_reexam_20261001_v1/scout_native/chain_round_stack_v4.sh)
- [code/opus_reexam_20261001_v1/scout_native/check_stack_vs_reference_v2.py](code/opus_reexam_20261001_v1/scout_native/check_stack_vs_reference_v2.py)
- [code/opus_reexam_20261001_v1/scout_native/extend_astra_compute_v1.py](code/opus_reexam_20261001_v1/scout_native/extend_astra_compute_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/fragment_astra_asz_v1.py](code/opus_reexam_20261001_v1/scout_native/fragment_astra_asz_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/fragment_astra_compute_v1.py](code/opus_reexam_20261001_v1/scout_native/fragment_astra_compute_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/fragment_astra_hzt_v2.py](code/opus_reexam_20261001_v1/scout_native/fragment_astra_hzt_v2.py)
- [code/opus_reexam_20261001_v1/scout_native/fragment_astra_stack1_v1.py](code/opus_reexam_20261001_v1/scout_native/fragment_astra_stack1_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/fragment_k4lf_v1.py](code/opus_reexam_20261001_v1/scout_native/fragment_k4lf_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/fragment_k4li_l1214_v1.py](code/opus_reexam_20261001_v1/scout_native/fragment_k4li_l1214_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/fragment_k4li_v1.py](code/opus_reexam_20261001_v1/scout_native/fragment_k4li_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/fragment_npx_v1.py](code/opus_reexam_20261001_v1/scout_native/fragment_npx_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/fragment_rows_v2.py](code/opus_reexam_20261001_v1/scout_native/fragment_rows_v2.py)
- [code/opus_reexam_20261001_v1/scout_native/fragment_sn3_v1.py](code/opus_reexam_20261001_v1/scout_native/fragment_sn3_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/make_idc_sn_v1.py](code/opus_reexam_20261001_v1/scout_native/make_idc_sn_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/run_cert_v2.py](code/opus_reexam_20261001_v1/scout_native/run_cert_v2.py)
- [code/opus_reexam_20261001_v1/scout_native/seal_alloc_sn_v1.py](code/opus_reexam_20261001_v1/scout_native/seal_alloc_sn_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/seal_judge_sn_v2.py](code/opus_reexam_20261001_v1/scout_native/seal_judge_sn_v2.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L2281](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2281)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2292](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2292)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2300](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2300)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2306](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2306)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2309](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2309)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2326](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2326)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2353](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2353)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2385](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2385)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2387](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2387)

<details>
<summary>All result files and records (3)</summary>

- [screen_decision_v1.json](screen_decision_v1.json) · record
- [confirm_decision_v1.json](confirm_decision_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Exact old tier gives -2.151% MSE; dropped pair differences are intrinsically high rank](../../../records/research/2026-10-11-exact-old-tier-gives-2-151-mse-dropped-pair/README.md)
- ← enables: [Native capacity retests reject rank 384 and rank-8 star: adjusted +0.378% and +0.454%](../native-capacity-retests-reject-rank-384-rank-8-star/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-rank-redistribution-rejected-restoring-layer-10-rank-384-gives) · [Copy provenance](../../../PROVENANCE.json)

