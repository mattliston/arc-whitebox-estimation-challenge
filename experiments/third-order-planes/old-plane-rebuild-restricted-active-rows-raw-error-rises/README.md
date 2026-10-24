# Old-plane rebuild restricted by active rows: raw error rises 3-39%

2026-09-23T03:15Z · experiment · rejected · measured

Restricting old-plane reconstruction to response or alive rows lost essential contributions. Lower frame ranks and gain-shaped recovery did not yield a useful one-third-cost representation.

Recorded measurement:

    2026-09-22 20:15 physicist: OLD TIER at 1/3 cost: NEGATIVE with arithmetic (oldtier/RESULT_OLDTIER.md). Tier = 107 dense units L5-14: rebuild 38%, inner 25%, join 26%, rotation 11%. Row-restricted rebuild NOT free (old diag response-only +30-39% raw; alive-only +3-7%; omitted-radial old part +2-6%). Frame rank 320/256/128: 1.0/2.5-2.8/9.5-11% plane error; lost content NOT gain-type (Sym(C x x) recovery removes 5-10%). Best: sub-frame age>=9@256 + rotation bookkeeping ~0.9x tier (~-3% bill) ~ wash vs raw.

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
<summary>Additional programs and dependencies (14)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_physicist_v1/model_bank_oldtier.py](code/fable_physicist_v1/model_bank_oldtier.py)
- [code/fable_physicist_v1/oldtier/old_sym_fit.py](code/fable_physicist_v1/oldtier/old_sym_fit.py)
- [code/fable_physicist_v1/oldtier/ot_cost_model.py](code/fable_physicist_v1/oldtier/ot_cost_model.py)
- [code/fable_physicist_v1/oldtier/trunc_gain_recover.py](code/fable_physicist_v1/oldtier/trunc_gain_recover.py)
- [code/fable_physicist_v1/ot_alive_ablate.py](code/fable_physicist_v1/ot_alive_ablate.py)
- [code/fable_physicist_v1/ot_consumer_ablate.py](code/fable_physicist_v1/ot_consumer_ablate.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L32](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L32)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT_OLDTIER.md](RESULT_OLDTIER.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-old-plane-rebuild-restricted-active-rows-raw-error-rises) · [Copy provenance](../../../PROVENANCE.json)

