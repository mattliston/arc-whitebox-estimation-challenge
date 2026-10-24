# Native routes add pair-map skew, kurtosis, kink-plane and per-layer correction dials

2026-10-13T00:10Z · tool · informational · measured

From candidate generation 54 (S54) the rounds could dial the pair map's local skew and kurtosis terms, a kink third-cumulant plane with index multiplicities (2,1) (K21) scale, the correction scale per layer and later the fourth-cumulant plane with index multiplicities (3,1) (K31) row factor and cohort-age local fourth-order term (K4L) ages 3-4 per band, each through a native route whose parity delivered D >= 0.5 before the port, with worker v15 / v16 certified bit for bit at zero strength.

Recorded measurement:

    pe3_12_14 D 0.997 / 0.995; pe4 0.998; k21k 0.998 / 0.997; ds_12_14 0.997 / 0.996; ds_15 1.000; c0_8_14 0.971 / 0.962; k4la3_8_15 0.938 / 0.958; k4la3_4_7 0.765 / 0.733; k4la4_4_7 FAILED D 0.28 / 0.26

## Programs

- [code/opus_reexam_20261001_v1/trainer/otr_m_v6.py](code/opus_reexam_20261001_v1/trainer/otr_m_v6.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl_v2.py](code/opus_reexam_20261001_v1/trainer/otr_pl_v2.py)
- [code/opus_reexam_20261001_v1/trainer/worker_generated_v15.py](code/opus_reexam_20261001_v1/trainer/worker_generated_v15.py)
- [code/opus_reexam_20261001_v1/trainer/worker_generated_v16.py](code/opus_reexam_20261001_v1/trainer/worker_generated_v16.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m_v7.py](code/opus_reexam_20261001_v1/trainer/otr_m_v7.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_pe34_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_pe34_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_k21k_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_k21k_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4l34_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4l34_v1.py)
- [code/opus_reexam_20261001_v1/trainer/export_dials_s46_v12.py](code/opus_reexam_20261001_v1/trainer/export_dials_s46_v12.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (41)</summary>

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
- [code/opus_reexam_20261001_v1/trainer/export.py](code/opus_reexam_20261001_v1/trainer/export.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1808](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1808)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1819](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1819)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1822](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1822)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1829](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1829)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1830](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1830)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1917](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1917)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1921](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1921)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1931](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1931)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1934](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1934)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1940](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1940)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2063](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2063)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2087](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2087)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2092](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2092)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2095](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2095)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2163](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2163)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2168](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2168)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2171](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2171)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2173](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2173)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Native-route acceptance requires correlation >= 0.70, size within 1/3..3 and delivery >= 0.5 on two networks](../../records/research/2026-10-12-native-route-acceptance-requires-correlation-0-70-size-within/README.md)
- enables: [Four-arm refit accepted: -0.137% independent and -0.231% sealed MSE change](../../records/research/2026-10-13-four-arm-refit-accepted-0-137-independent-0-231/README.md)
- enables: [Hub quartic ages 3-4 extended to layers 8-15: -0.513% mean-log MSE on 24 networks](../../records/research/2026-10-13-hub-quartic-ages-3-4-extended-layers-8-15/README.md)
- enables: [Pair-map skew and response-row regression variants fail fresh or out-of-selection checks](../../experiments/covariance-structure/pair-map-skew-response-row-regression-variants-fail-fresh/README.md)

[Research graph](../../graph/index.html#node=2026-10-13-native-routes-add-pair-map-skew-kurtosis-kink-plane) · [Copy provenance](../../PROVENANCE.json)

