# Late-only rank-8 star rejected: +0.150% MSE and about +0.15% FLOPs

2026-10-12T17:44:55Z · experiment · rejected · measured

The late-capacity idea that the all-layer forms' gains sit late was tested: at the candidate generation 50 (S50) frozen point the moment-closure estimator's strength-scaled one-direction star was replaced at layers 11-14 by star0 + beta (star8 - star0) (control K = -1 reproduced the base within 6.0e-5 per layer); on the 4 networks with a base the final-layer MSE was +0.150% pooled (lower on 2 of 4; L11-14 -0.07 / -0.07 / -0.04 / +0.00%, L15 +0.15%) against the engineer's census price of +0.15% metered FLOPs (the star's per-layer cost is nearly flat, 0.63% of the metered FLOPs per layer), so no net gain is possible. The pair-mean truncation pair-mean read rank (r_u) 320 at layers 11-14 only was priced at about +0.80% metered FLOPs (58-70% of the all-layer +1.26%) and deferred.

Recorded measurement:

    ss 8 at L11-14: final +0.150% pooled (mean log +0.110%, se 0.272, lower on 2 of 4), all-layer +0.00%, L11-14 -0.07 / -0.07 / -0.04 / +0.00%, L15 +0.15%; bill +0.15%, about +50 ops; r_u 320 at L11-14: about +0.80% bill (+0.73..+0.88%), +250-300 ops

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/physicist/t131_worker_ss8late.py](code/opus_reexam_20261001_v1/physicist/t131_worker_ss8late.py)
- [code/opus_reexam_20261001_v1/physicist/t131_chain.sh](code/opus_reexam_20261001_v1/physicist/t131_chain.sh)
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
<summary>Additional programs and dependencies (42)</summary>

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
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/jmS50_v1tw_x/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/jmS50_v1tw_x/candidate.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/t129_worker_floors_v12.py](code/opus_reexam_20261001_v1/physicist/t129_worker_floors_v12.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m_v5.py](code/opus_reexam_20261001_v1/trainer/otr_m_v5.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)
- [code/opus_reexam_20261001_v1/trainer/worker_generated_v12.py](code/opus_reexam_20261001_v1/trainer/worker_generated_v12.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1534](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1534)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1535](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1535)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1540](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1540)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1591](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1591)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1595](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1595)

<details>
<summary>All result files and records (37)</summary>

- [STOP](STOP) · result
- [allowed_seeds.txt](allowed_seeds.txt) · result
- [t131_ss8late_7100136.w0.json](t131_ss8late_7100136.w0.json) · result
- [t131_ss8late_7100137.w1.json](t131_ss8late_7100137.w1.json) · result
- [t131_ss8late_7100138.w0.json](t131_ss8late_7100138.w0.json) · result
- [t131_ss8late_7100139.w1.json](t131_ss8late_7100139.w1.json) · result
- [t131_ss8late_7100140.w0.json](t131_ss8late_7100140.w0.json) · result
- [t131_ss8late_7100141.w1.json](t131_ss8late_7100141.w1.json) · result
- [t131_ss8late_7100142.w0.json](t131_ss8late_7100142.w0.json) · result
- [t131_ss8late_7100143.w1.json](t131_ss8late_7100143.w1.json) · result
- [t131_ss8late_7100144.w0.json](t131_ss8late_7100144.w0.json) · result
- [t131_ss8late_7100145.w1.json](t131_ss8late_7100145.w1.json) · result
- [t131_ss8late_7100146.w0.json](t131_ss8late_7100146.w0.json) · result
- [t131_ss8late_7100147.w1.json](t131_ss8late_7100147.w1.json) · result
- [t131_ss8late_7100148.w0.json](t131_ss8late_7100148.w0.json) · result
- [t131_ss8late_7100149.w1.json](t131_ss8late_7100149.w1.json) · result
- [t131_ss8late_7100150.w0.json](t131_ss8late_7100150.w0.json) · result
- [t131_ss8late_7100151.w1.json](t131_ss8late_7100151.w1.json) · result
- [t131_ss8late_7100136.json](t131_ss8late_7100136.json) · result
- [t131_ss8late_7100137.json](t131_ss8late_7100137.json) · result
- [t131_ss8late_7100138.json](t131_ss8late_7100138.json) · result
- [t131_ss8late_7100139.json](t131_ss8late_7100139.json) · result
- [t131_ss8late_7100140.json](t131_ss8late_7100140.json) · result
- [t131_ss8late_7100141.json](t131_ss8late_7100141.json) · result
- [t131_ss8late_7100142.json](t131_ss8late_7100142.json) · result
- [t131_ss8late_7100143.json](t131_ss8late_7100143.json) · result
- [t131_ss8late_7100144.json](t131_ss8late_7100144.json) · result
- [t131_ss8late_7100145.json](t131_ss8late_7100145.json) · result
- [t131_ss8late_7100146.json](t131_ss8late_7100146.json) · result
- [t131_ss8late_7100147.json](t131_ss8late_7100147.json) · result
- [t131_ss8late_7100148.json](t131_ss8late_7100148.json) · result
- [t131_ss8late_7100149.json](t131_ss8late_7100149.json) · result
- [t131_ss8late_7100150.json](t131_ss8late_7100150.json) · result
- [worker_w0.log](worker_w0.log) · log
- [worker_w1.log](worker_w1.log) · log
- [census_layer_s50tw.json](../../flop-accounting/flop-census-two-winograd-leaf-products-consume-67-float64/census_layer_s50tw.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-late-only-rank-8-star-rejected-0-150-mse) · [Copy provenance](../../../PROVENANCE.json)

