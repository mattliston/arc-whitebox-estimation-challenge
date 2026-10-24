# Cut-simulated research estimator removes 95% of the native prediction discrepancy

2026-10-11T00:16Z · tool · informational · derived

The advertised cut simulator was stale (no pair-mean truncation, wrong hook order); a composite rebuilt from the native joins, row cuts, local pair-mean read rank (r_u) 256, radial repair and cohort-age local fourth-order term (K4L) projection failed at first (10.5x residual power from L7) until the law's response-row regression, the uncut old right factor and the beta column were made native-faithful.

Recorded measurement:

    v7 ON 651 residual RMS 5.65e-5 vs original 1.75e-5 (10.49x power); v11 all three law flags true: 3.74e-6 / 3.87e-6 on 651 / 378, residual power 0.046 / 0.051 (95.4% / 94.9% removed)

## Programs

- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_v2.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
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
<summary>Additional programs and dependencies (32)</summary>

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
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/physicist/t86_local_cut_vjp_v5.py](code/opus_reexam_20261001_v1/physicist/t86_local_cut_vjp_v5.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L730](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L730)
- [TEAM_LOG_2026-10-01_to_10-22.md#L740](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L740)
- [TEAM_LOG_2026-10-01_to_10-22.md#L741](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L741)
- [TEAM_LOG_2026-10-01_to_10-22.md#L743](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L743)
- [TEAM_LOG_2026-10-01_to_10-22.md#L756](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L756)
- [TEAM_LOG_2026-10-01_to_10-22.md#L758](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L758)
- [TEAM_LOG_2026-10-01_to_10-22.md#L762](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L762)
- [TEAM_LOG_2026-10-01_to_10-22.md#L771](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L771)
- [TEAM_LOG_2026-10-01_to_10-22.md#L773](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L773)
- [TEAM_LOG_2026-10-01_to_10-22.md#L775](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L775)
- [TEAM_LOG_2026-10-01_to_10-22.md#L777](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L777)
- [TEAM_LOG_2026-10-01_to_10-22.md#L779](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L779)
- [TEAM_LOG_2026-10-01_to_10-22.md#L793](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L793)
- [TEAM_LOG_2026-10-01_to_10-22.md#L832](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L832)
- [TEAM_LOG_2026-10-01_to_10-22.md#L838](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L838)
- [TEAM_LOG_2026-10-01_to_10-22.md#L789](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L789)
- [TEAM_LOG_2026-10-01_to_10-22.md#L816](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L816)
- [TEAM_LOG_2026-10-01_to_10-22.md#L830](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L830)

<details>
<summary>All result files and records (6)</summary>

- [S46_v11_PINNED_PROFILE.json](S46_v11_PINNED_PROFILE.json) · record
- [S46_CERTIFICATION_PLAN_v2.md](S46_CERTIFICATION_PLAN_v2.md) · explainer
- [CERTIFICATE_V11_SCOPED_v1.json](CERTIFICATE_V11_SCOPED_v1.json) · record
- [localization_651_v1.json](localization_651_v1.json) · result
- [T86_LOCAL_VJP_REVIEW_v1.json](T86_LOCAL_VJP_REVIEW_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Three native approximations explain 91% of the native-versus-research error gap](../../records/research/2026-10-10-three-native-approximations-explain-91-native-versus-research-error/README.md)
- enables: [First cut-simulated refit rejected: independent MSE change -0.235%, standard error 0.340](../../experiments/learned-correction/first-cut-simulated-refit-rejected-independent-mse-change-0/README.md)

[Research graph](../../graph/index.html#node=2026-10-11-cut-simulated-research-estimator-removes-95-native-prediction-discrepancy) · [Copy provenance](../../PROVENANCE.json)

