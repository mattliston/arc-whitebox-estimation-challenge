# Common-factor covariance model rejected: +0.200% MSE for +1.6..2.4% source FLOPs

2026-10-13T08:30Z · experiment · rejected · measured

The independent hunt's never-run arm, a compensated common mean / variance factor of the covariance (CFCOV, a 5-node latent quadrature at strength 1.0 on layers 8-14), was screened at the candidate generation 51 (S51) + transport-gate point against the physicist's controls on 8 fresh networks: it is worse with no metered FLOPs credit and far from its pre-declared -2% bar, so with the radial covariance model also null, resumming the radial / common factor exactly does not pay at the current moment-closure estimator (the learned parts already hold it). The first control was killed by the launcher's 4 GB RSS cap (RSS 4.10 GB) and re-run at 7.5 GB.

Recorded measurement:

    arm vs t134 control final-layer MSE mean log +0.200% (se 0.169), pooled +0.232%, lower on 2 of 8 (per network +0.49 +0.32 +0.41 +0.41 -0.84 -0.18 +0.50 +0.50%); pooled by layer L9-15 +0.01 / +0.06 / +0.06 / +0.04 / +0.23 / +0.23 / +0.23%; source bill +1.6..2.4%; screen bar mean log <= -2%; control matched on all 16 layer MSEs

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_hunt_20261013_v1/harness_audit/launch_cfcov_v1b.sh](code/astra_hunt_20261013_v1/harness_audit/launch_cfcov_v1b.sh)
- [code/astra_hunt_20261013_v1/harness_audit/astra_runtime_safety_v1.py](code/astra_hunt_20261013_v1/harness_audit/astra_runtime_safety_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/compare_cfcov_v1.py](code/astra_hunt_20261013_v1/harness_audit/compare_cfcov_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/launch_cfcov_v1.sh](code/astra_hunt_20261013_v1/harness_audit/launch_cfcov_v1.sh)
- [code/astra_hunt_20261013_v1/harness_audit/otr_cut_runtime_astra_v1.py](code/astra_hunt_20261013_v1/harness_audit/otr_cut_runtime_astra_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/prepare_cfcov_v1.py](code/astra_hunt_20261013_v1/harness_audit/prepare_cfcov_v1.py)
- [code/astra_hunt_20261013_v1/harness_audit/screen_entry_v2.py](code/astra_hunt_20261013_v1/harness_audit/screen_entry_v2.py)
- [code/astra_hunt_20261013_v1/harness_audit/worker_astra_v1.py](code/astra_hunt_20261013_v1/harness_audit/worker_astra_v1.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cfcov_toy_cert_v1.py](code/astra_hunt_20261013_v1/physics_alternative/cfcov_toy_cert_v1.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_cfcov_v1/n6y_cutsim_v2.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_cfcov_v1/n6y_cutsim_v2.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_cfcov_v1/pbase_cutsim_v1.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_cfcov_v1/pbase_cutsim_v1.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_cfcov_v1/torch_cutsim_s46_v11.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_cfcov_v1/torch_cutsim_s46_v11.py)

<details>
<summary>Additional programs and dependencies (55)</summary>

- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_cfcov_v1/torch_push_native_v2.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_cfcov_v1/torch_push_native_v2.py)
- [code/astra_hunt_20261013_v1/physics_alternative/cutsim_cfcov_v1/torch_rowcuts_v15.py](code/astra_hunt_20261013_v1/physics_alternative/cutsim_cfcov_v1/torch_rowcuts_v15.py)
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
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc1/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/make_k3_v7.py](code/opus_reexam_20261001_v1/scout_card/make_k3_v7.py)
- [code/opus_reexam_20261001_v1/scout_card/make_rb_v1.py](code/opus_reexam_20261001_v1/scout_card/make_rb_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L2016](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2016)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2017](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2017)

<details>
<summary>All result files and records (2)</summary>

- [cfcov_arm_guard_v1b.json.comparison.json](cfcov_arm_guard_v1b.json.comparison.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-common-factor-covariance-model-rejected-0-200-mse-1) · [Copy provenance](../../../PROVENANCE.json)

