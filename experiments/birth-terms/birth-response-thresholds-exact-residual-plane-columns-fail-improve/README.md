# Birth and response thresholds and exact residual-plane columns fail to improve adjusted error

2026-09-23T04:41Z · experiment · rejected · measured

Threshold changes increased error or column count; birth clamps were already lossless. Representing the exact distinct star residual required too many columns for its small raw gain.

Recorded measurement:

    2026-09-22 21:41 physicist: BOOKKEEPING AUDIT (fable_physicist_v1/audit/RESULT_AUDIT.md; torch, learned ON, no refit, 35/100/37/39). Only the old-tier frame pays (no-frame oracle +1.7/-4.0/-2.7/-1.5%, mean -1.6%). Select thresholds co-adapted at .01 both ways: births .001 mean -1.06% (mixed, +12% cols), response .001 +2.1%, alive .001 +13%, all 1e-5 +90%; births .02/.03 +3.5/+15%. Birth clamps lossless (u never; gap floor 10-19% of births -> star diagonal, exact). Star distinct-part loss: residual as exact CP columns = -2.3/+0.2/0.0/-0.6% (mean -0.7%) for ~3.6k cols/layer: dead.

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
<summary>Additional programs and dependencies (11)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_physicist_v1/audit_run.py](code/fable_physicist_v1/audit_run.py)
- [code/fable_physicist_v1/model_bank_audit.py](code/fable_physicist_v1/model_bank_audit.py)
- [code/fable_physicist_v1/star_cols_oracle.py](code/fable_physicist_v1/star_cols_oracle.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L53](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L53)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT_AUDIT.md](RESULT_AUDIT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-birth-response-thresholds-exact-residual-plane-columns-fail-improve) · [Copy provenance](../../../PROVENANCE.json)

