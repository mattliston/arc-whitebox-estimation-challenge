# Whole-tier rank 256 or 288 with joint refit does not recover the lost accuracy

2026-09-23T01:00Z · experiment · rejected · measured

Lowering the shared old-tier frame to rank 256 (native util about -10..-11.5%, raw +38-43% before refit) or 288 and jointly refitting the learned closure did not recover the lost content: the rank-256 training loss stayed at 2.60e-8 after 28 updates and the rank-288 run's epoch-0 VALID16 got worse than its start.

Recorded measurement:

    VALID8 before refit: rank 256 2.5444e-8, rank 288 2.2602e-8 vs control 1.8893e-8; refit_rank256_lr1e4_w20 train 2.5958e-8 at update 28 (killed, no recovery); refit_rank288_full_lr2e4 epoch 0 VALID16 2.2484e-8 -> 2.3609e-8 (+5.0%, 0/16 improved) vs break-even 2.030e-8

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)

<details>
<summary>Additional programs and dependencies (13)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_refit_v1/cand_rank256.py](code/fable_refit_v1/cand_rank256.py)
- [code/fable_refit_v1/cand_rank288.py](code/fable_refit_v1/cand_rank288.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/fable_refit_v1/refit.py](code/fable_refit_v1/refit.py)
- [code/fable_refit_v1/score_variant.py](code/fable_refit_v1/score_variant.py)

</details>

## Evidence

- [NOTES.md#L177](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L177)
- [TEAM_LOG_2026-09-22_to_09-26.md#L157](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L157)

<details>
<summary>All result files and records (7)</summary>

- [refit_rank256_lr1e4_w20.log](refit_rank256_lr1e4_w20.log) · log
- [refit_rank288_full_lr2e4.log](refit_rank288_full_lr2e4.log) · log
- [calib_rank256.log](calib_rank256.log) · log
- [calib_rank288.log](calib_rank288.log) · log
- [chain_rank256.log](chain_rank256.log) · log
- [NOTES.md](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-whole-tier-rank-256-or-288-joint-refit-does) · [Copy provenance](../../../PROVENANCE.json)

