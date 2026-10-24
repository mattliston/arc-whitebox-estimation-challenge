# Birth pruning, low-rank young reads, covariance-channel corrections and edge inputs fail measured gates

2026-10-11T11:35Z · experiment · rejected · measured

Every pre-gated bet of 10-11 failed: pair energies are flat and mutually cancelling, the young plane is not low rank, the moment-closure estimator already holds 97-99% of the true scale-mode coefficient (gamma 1 doubled it: +287%), per-row fourth-cumulant plane with index multiplicities (3,1) (K31) factors are the existing pair-map third-cumulant coefficient (k31) dial, and no new per-unit edge input moves the population.

Recorded measurement:

    99.9% of pair energy needs 97.4-99.9% of pairs; rank-384 read keeps 16-24% error at L2-6; K21_C gamma 0.5 +0.20%, gamma 1 +287%; K31 per-row models remove 12-17% of error on 100; edge inputs best t^2 2.42 vs null ~1.0

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/engineer/a61_prune/census_v1.py](code/opus_reexam_20261001_v1/engineer/a61_prune/census_v1.py)
- [code/opus_reexam_20261001_v1/physicist/t98_worker_k21c.py](code/opus_reexam_20261001_v1/physicist/t98_worker_k21c.py)
- [code/opus_reexam_20261001_v1/physicist/t100_k21_decomp.py](code/opus_reexam_20261001_v1/physicist/t100_k21_decomp.py)
- [code/opus_reexam_20261001_v1/physicist/t99_k31_gate.py](code/opus_reexam_20261001_v1/physicist/t99_k31_gate.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)

<details>
<summary>Additional programs and dependencies (33)</summary>

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
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_m.py](code/opus_reexam_20261001_v1/trainer/otr_m.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)
- [code/opus_reexam_20261001_v1/trainer/worker_generated_v4.py](code/opus_reexam_20261001_v1/trainer/worker_generated_v4.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1080](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1080)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1081](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1081)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1082](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1082)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1083](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1083)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1085](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1085)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1086](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1086)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1087](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1087)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1089](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1089)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1090](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1090)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1091](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1091)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [edge_table_v1.out](edge_table_v1.out) · result

</details>

## Related work

- ← enables: [True plane inputs at layers 10-14 give -17.5..-19.5% MSE; available carrier changes cost about their gains](../../../records/research/2026-10-10-true-plane-inputs-layers-10-14-give-17-5/README.md)
- ← enables: [Repeated public and generated gradient refits converge to the existing point](../../../records/research/2026-10-11-repeated-public-generated-gradient-refits-converge-existing-point/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-birth-pruning-low-rank-young-reads-covariance-channel-corrections) · [Copy provenance](../../../PROVENANCE.json)

