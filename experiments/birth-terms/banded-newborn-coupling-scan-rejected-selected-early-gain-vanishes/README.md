# Banded newborn coupling scan rejected: selected early gain vanishes on 24 networks

2026-10-13T03:16Z · experiment · rejected · measured

The physicist's three-band beta scan (gbs = the newborn hub's leg scaled) on top of the transport gate, paired against it on 8 fresh networks: the +1 band at layers 10-11 and the -0.5 band at layers 12-14 both read slightly worse; only the -0.5 band at layers 5-9 read better on 8 and was extended to 24, where it vanished.

Recorded measurement:

    gbsp1L10_11: pooled +0.063%, mean log +0.051% (se 0.067, 0.8 SE), 3 of 8 lower; gbsm05L12_14: +0.068% / +0.072% (se 0.040, 1.8 SE); gbsm05L5_9: -0.496% / -0.421% (se 0.337), 6 of 8; at n = 24 +0.028% / +0.001% (se 0.201), extension 16 alone +0.283% / +0.212%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/scout_card/auto_k14_v2.sh](code/opus_reexam_20261001_v1/scout_card/auto_k14_v2.sh)
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

<details>
<summary>Additional programs and dependencies (47)</summary>

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
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc6/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc6/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc6/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc6/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc6/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc6/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc6/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc6/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc6/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc6/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc7/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc7/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc7/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc7/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc7/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc7/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc7/torch_push_native_v2.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc7/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/cutsim_sc7/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/scout_card/cutsim_sc7/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/scout_card/k_line_v1.py](code/opus_reexam_20261001_v1/scout_card/k_line_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/make_k14_v2.py](code/opus_reexam_20261001_v1/scout_card/make_k14_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_cut_runtime_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_cut_runtime_sc4.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py)
- [code/opus_reexam_20261001_v1/scout_card/worker_sc8.py](code/opus_reexam_20261001_v1/scout_card/worker_sc8.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L1899](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1899)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1893](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1893)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1894](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1894)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1914](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1914)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-banded-newborn-coupling-scan-rejected-selected-early-gain-vanishes) · [Copy provenance](../../../PROVENANCE.json)

