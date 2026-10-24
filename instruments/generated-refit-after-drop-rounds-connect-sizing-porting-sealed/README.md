# Generated refit-after-drop rounds connect sizing, porting, sealed judging and release through verdict files

2026-10-12T08:11Z · tool · informational · measured

After candidate generation 50 (S50) the refit-after-drop rounds were generated from a template (seed blocks per round, sizing / check / sealed / reserve), each chain waiting on the previous round's STOP; the engineer's port loop, the physicist's stage-2 loop and the release loop keyed on verdict files and anchored TEAM_LOG lines, with lead promotion and a continue rule (plus1se) for the start point.

Recorded measurement:

    rounds S55-S63 armed 10-12 08:03Z, S64-S66 11:14Z; block = 6 analyst blocks (phase 1 blocks 1+3, check / sealed block 2, reserve 5); continue rule plus1se from S53

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/gen_round.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/gen_round.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v5.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/maintain_workers_v5.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/lead_promote_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/lead_promote_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/fix_chain_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/fix_chain_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/chain_rad_port_v9.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/chain_rad_port_v9.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/chain_rad_port_v11.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/chain_rad_port_v11.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t127_stage2_loop_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t127_stage2_loop_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t127_stage2_loop_v12.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t127_stage2_loop_v12.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_loop_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_loop_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_loop_v3.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_loop_v3.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)

<details>
<summary>Additional programs and dependencies (97)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_analyst_v1/s2card.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_analyst_v1/s2card.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_analyst_v1/s6bcard.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_analyst_v1/s6bcard.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_covhunter_v1/covstep.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_covhunter_v1/covstep.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_covhunter_v1/covstep_s2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_covhunter_v1/covstep_s2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_covhunter_v1/gn3_knobs.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_covhunter_v1/gn3_knobs.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_covhunter_v1/s6bcard_joint2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_covhunter_v1/s6bcard_joint2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_covhunter_v1/s6bcard_joint2t.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_covhunter_v1/s6bcard_joint2t.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1/gn_linear_hook.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1/gn_linear_hook.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1/join_nystrom_torch.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1/join_nystrom_torch.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1/pair_series_torch.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1/pair_series_torch.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_physicist_v1/refit6b/s6bcard_loop5js.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1/star_nys_test.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1/star_nys_test.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/model_bank_bfix.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/model_bank_bfix.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/model_bank_research.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1/model_bank_research.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a41_base/fragment_gsk_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a41_base/fragment_gsk_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/gate_rdz_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/gate_rdz_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/gk_guard_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/gk_guard_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_gsk_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_gsk_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_xsg_any_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_xsg_any_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/parity_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/parity_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s48/m_sd_rc/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s48/m_sd_rc/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/quick.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/quick.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/gate.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/gate.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/b6cols.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/b7cols.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/f4b.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/pbase_frozen_s4a.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/pbase_frozen_s4a.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t112_chain_v15.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t112_chain_v15.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t112_chain_v17.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t112_chain_v17.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t112_table.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t112_table.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t85_pinned_judge_v14.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t85_pinned_judge_v14.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_client.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_client.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_run.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/build_variant.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/build_variant.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/designation_judge.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/designation_judge.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/own_gate.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/own_gate.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/project.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/project.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v4.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v4.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/compare_twin_native_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/compare_twin_native_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/rusage.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/rusage.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/s47_safe_v7.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/s47_safe_v7.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/server_light.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/server_light.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/setup_clock.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_a_package.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_a_package.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_c_boundary.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_c_boundary.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/twin_pipeline_v3.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/twin_pipeline_v3.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/twin_pipeline_v5.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/twin_pipeline_v5.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/S54/chain_s54.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/S54/chain_s54.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/S54/hold_s54.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/S54/hold_s54.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/S54/make_s54.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/S54/make_s54.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/launch_gen_v10.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/launch_gen_v10.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/launch_gen_v10_t2.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/launch_gen_v10_t2.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/make_exportable.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/make_exportable.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_v7.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_v7.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/export.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/export.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/export_dials_s46_v6.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/export_dials_s46_v6.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_engine.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_m.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_m_v5.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_m_v5.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_pl.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_pl.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/worker_generated_v12.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/worker_generated_v12.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1392](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1392)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1394](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1394)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1417](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1417)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1419](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1419)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1423](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1423)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1433](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1433)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1436](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1436)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1438](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1438)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1467](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1467)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1546](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1546)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1551](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1551)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1556](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1556)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1562](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1562)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1780](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1780)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1785](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1785)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [CONTINUE_RULE](CONTINUE_RULE) · record

</details>

## Related work

- enables: [Combined-candidate workflow applies verified changes, seals on fresh networks and runs a fast release gate](../../records/research/2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/README.md)
- enables: [Re-sizing previously fitted arms gives no fresh-network signal; two-stage pruning fails its stability rule](../../experiments/learned-correction/re-sizing-previously-fitted-arms-gives-no-fresh-network/README.md)
- ← enables: [Refit-after-drop strength round accepted: -0.203% pooled check and -0.379% sealed MSE](../../records/research/2026-10-12-refit-after-drop-strength-round-accepted-0-203-pooled/README.md)

[Research graph](../../graph/index.html#node=2026-10-12-generated-refit-after-drop-rounds-connect-sizing-porting-sealed) · [Copy provenance](../../PROVENANCE.json)

