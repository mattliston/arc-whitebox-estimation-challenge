# Early plane read restored to full rank: fresh-16 MSE -0.622% for bill +1.225%, net +0.604%, rejected

2026-10-14T00:09:07Z · experiment · rejected · measured

Reading the carried plane bank at the early layers without the rank cut (the 'early-read full' arm) bought accuracy at twice its price in bill, so the arm was closed; smaller source-level savings were pursued separately.

Recorded measurement:

    complete fresh 16: MSE log -0.621564750%, frozen bill +1.225440016%, net +0.603875266%

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
<summary>Additional programs and dependencies (30)</summary>

- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/candidate_early_vread_noop_UNVALIDATED_astra_v1_v3.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/candidate_early_vread_noop_UNVALIDATED_astra_v1_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/candidate_early_vread_on_UNVALIDATED_astra_v1_v3.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/candidate_early_vread_on_UNVALIDATED_astra_v1_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/runtime_early_vread_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/runtime_early_vread_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/worker_early_vread_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/capacity_early_vread_held_astra_v1_v3/worker_early_vread_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/launch_early_vread_astra_v4.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/launch_early_vread_astra_v4.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/prepare_early_vread_astra_v3.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/prepare_early_vread_astra_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/prepare_early_vread_astra_v4.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/prepare_early_vread_astra_v4.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/run_native_price_early_vread_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/run_native_price_early_vread_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/run_native_price_early_vread_astra_v2.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/run_native_price_early_vread_astra_v2.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/table_early_vread_astra_v3.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/table_early_vread_astra_v3.py)
- [code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/table_early_vread_astra_v4.py](code/opus_reexam_20261001_v1/scout_card/early_vread_protocol_held_astra_v1/table_early_vread_astra_v4.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py)
- [code/opus_reexam_20261001_v1/scout_card/run_rank_native_price_astra_v1.py](code/opus_reexam_20261001_v1/scout_card/run_rank_native_price_astra_v1.py)
- [code/opus_reexam_20261001_v1/scout_native/run_cert_v2.py](code/opus_reexam_20261001_v1/scout_native/run_cert_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2438](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2438)

<details>
<summary>All result files and records (2)</summary>

- [INDEPENDENT_SOURCE_REVIEW_v1.json](INDEPENDENT_SOURCE_REVIEW_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-early-plane-read-restored-full-rank-fresh-16-mse) · [Copy provenance](../../../PROVENANCE.json)

