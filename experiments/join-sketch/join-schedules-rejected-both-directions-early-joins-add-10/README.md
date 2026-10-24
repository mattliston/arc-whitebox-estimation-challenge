# Join schedules rejected in both directions: early joins add 10.67% MSE; later joins cost more than they gain

2026-10-11T20:23Z · experiment · rejected · measured

The submitted joins 7/10/13 (absorption at ages 5-7) sit just past a cliff: giving up one more layer of youth is catastrophic, adding one buys almost nothing for its metered FLOPs; each layer of delay cuts the lost sketch energy 3x but 1e-4 of the variance is visible in the final error.

Recorded measurement:

    early 6/9/12 +10.67% (13 of 13 worse) for -4.51% bill; 7/10/- -0.232% (se 0.129) for +0.65% (native +1.39..+1.64%); 7/11/- +1.125%; 8/11/14 -1.305% for +4.18% bill

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/physicist/t115_worker_joins.py](code/opus_reexam_20261001_v1/physicist/t115_worker_joins.py)
- [code/opus_reexam_20261001_v1/physicist/t116_table_v2.py](code/opus_reexam_20261001_v1/physicist/t116_table_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/price_joins_v2.py](code/opus_reexam_20261001_v1/engineer/a62_modules/price_joins_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_join710_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_join710_v1.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)

<details>
<summary>Additional programs and dependencies (30)</summary>

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
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/table_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/table_v1.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1165](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1165)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1167](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1167)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1173](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1173)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1179](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1179)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1180](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1180)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1184](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1184)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1190](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1190)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1191](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1191)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1192](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1192)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Cheaper young-plane carriers rejected: earlier joins add 40-48% MSE; rank-256 reads add 131%](../../third-order-planes/cheaper-young-plane-carriers-rejected-earlier-joins-add-40/README.md)
- enables: [Exact old tier gives -2.151% MSE; dropped pair differences are intrinsically high rank](../../../records/research/2026-10-11-exact-old-tier-gives-2-151-mse-dropped-pair/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-join-schedules-rejected-both-directions-early-joins-add-10) · [Copy provenance](../../../PROVENANCE.json)

