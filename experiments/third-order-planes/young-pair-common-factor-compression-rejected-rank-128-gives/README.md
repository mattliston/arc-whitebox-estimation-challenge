# Young-pair common-factor compression rejected: rank 128 gives +18.41% research MSE and +91% native MSE

2026-10-12T04:08Z · experiment · rejected · derived

Each young pair p+- = c +- h splits exactly; projecting only the right factor's c on the top-384 eigenvectors costs +0.16%, but compressing c everywhere (B384) costs +13.5% and the read saves metered FLOPs only below rank ~380; the oracle rank-128 basis costs +18% and the native per-layer Nystrom r 128 gave -1.64% metered FLOPs for +91% MSE.

Recorded measurement:

    A384 +0.156% (se 0.202); B384 +13.52% (0 of 15); A128 +18.41% (0 of 12); native psrN128 bill -1.64%, requests +1,535, final MSE +91%; exact split psr0 = S48b to rounding at +0.37% bill

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/physicist/t123_worker_pairsplit.py](code/opus_reexam_20261001_v1/physicist/t123_worker_pairsplit.py)
- [code/opus_reexam_20261001_v1/physicist/t123_table.py](code/opus_reexam_20261001_v1/physicist/t123_table.py)
- [code/opus_reexam_20261001_v1/physicist/t124_worker_rbasis_v2.py](code/opus_reexam_20261001_v1/physicist/t124_worker_rbasis_v2.py)
- [code/opus_reexam_20261001_v1/physicist/t125_worker_nys.py](code/opus_reexam_20261001_v1/physicist/t125_worker_nys.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_psr_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_psr_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_psr_v3.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_psr_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/probe_ycbasis_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/probe_ycbasis_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/psrN128/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/psrN128/candidate.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)

<details>
<summary>Additional programs and dependencies (35)</summary>

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
- [code/opus_reexam_20261001_v1/physicist/t123_common.py](code/opus_reexam_20261001_v1/physicist/t123_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1272](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1272)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1274](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1274)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1275](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1275)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1279](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1279)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1283](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1283)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1284](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1284)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1288](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1288)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1289](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1289)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1291](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1291)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1292](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1292)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1293](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1293)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1294](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1294)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1295](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1295)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1297](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1297)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1299](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1299)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1301](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1301)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1302](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1302)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1304](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1304)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1307](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1307)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1308](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1308)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1309](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1309)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Cheaper young-plane carriers rejected: earlier joins add 40-48% MSE; rank-256 reads add 131%](../cheaper-young-plane-carriers-rejected-earlier-joins-add-40/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-young-pair-common-factor-compression-rejected-rank-128-gives) · [Copy provenance](../../../PROVENANCE.json)

