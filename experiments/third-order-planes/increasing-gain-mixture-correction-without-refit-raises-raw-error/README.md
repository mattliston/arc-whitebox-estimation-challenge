# Increasing the gain-mixture correction without refit raises raw error 8–52%

2026-09-23T01:14Z · experiment · rejected · measured

Repartitioning the gain-mixture content into the carried b vector at full physical strength makes the final MSE worse on all four research networks, and a physics-only card (GNN hook and edges zeroed) degrades too, so the star re-absorbs what b receives and the plane residual is unchanged; an additive b variant makes the plane right but explodes the means 10-20x because the full Sym(C,m) tensor also adds 3 v kappa mu to each unit's k3, which the truth does not have.

Recorded measurement:

    learned ON (4 nets): gamma .5 +8%, .85 +34%, 1.0 +52%; physics-only (both_off): r35 2.038e-8 -> 2.336e-8 (gamma .5) -> 3.257e-8 (1.0), r100 2.157e-8 -> 2.818e-8 -> 4.601e-8; additive b gamma .5: plane rel err L8/9/12/13 .60/.63/.62/.60 -> .47/.47/.44/.44 but L8 mean error 6.8e-9 -> 1.6e-7

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_refit_v1/bfix_test.py](code/fable_refit_v1/bfix_test.py)
- [code/fable_refit_v1/bfix_physics_only_test.py](code/fable_refit_v1/bfix_physics_only_test.py)
- [code/fable_refit_v1/bfix_additive_test.py](code/fable_refit_v1/bfix_additive_test.py)
- [code/fable_refit_v1/bfix_mode_test.py](code/fable_refit_v1/bfix_mode_test.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)

<details>
<summary>Additional programs and dependencies (13)</summary>

- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
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
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)

</details>

## Evidence

- [NOTES.md#L175](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L175)
- [NOTES.md#L178](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L178)
- [NOTES.md#L180](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L180)

<details>
<summary>All result files and records (5)</summary>

- [bfix_test_m.log](bfix_test_m.log) · log
- [bfix_physics_only_test.log](bfix_physics_only_test.log) · log
- [bfix_additive_test.log](bfix_additive_test.log) · log
- [bfix_offb.log](bfix_offb.log) · log
- [NOTES.md](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

## Related work

- measured by: [Rank-one covariance–mean gain mixture explains 63–68% of the (2,1)-plane error](../../../records/research/2026-09-23-rank-one-covariancemean-gain-mixture-explains-6368-2-1/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-increasing-gain-mixture-correction-without-refit-raises-raw-error) · [Copy provenance](../../../PROVENANCE.json)

