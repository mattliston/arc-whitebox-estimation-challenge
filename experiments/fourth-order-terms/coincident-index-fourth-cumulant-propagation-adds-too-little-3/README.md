# Coincident-index fourth-cumulant propagation adds too little to the (3,1) plane

2026-09-23T03:59Z · experiment · rejected · measured

A rectified-Gaussian fourth-cumulant push over coincident indices is 2-6% of the true K31 scale and adds at most 0.7 points of R^2; projecting supplies onto the mean-orthogonal subspace changes nothing for K31 (already orthogonal). The companion covariance-only oracle shows the three truth planes together give the whole covariance lever (.762) while truth K21 alone hurts (1.056), because the card's K21 error compensates its fourth-order errors.

Recorded measurement:

    coincident-index K4 push 2-6% of truth K31 scale, <= 0.7 pt R^2; isotropic forms explain K22 94-98%, per-unit k4 94-98%, K31 40-50%; planes cov-only perp: K21+K22+K31 .762, K31 alone .874, K22+K31 .844, K21 alone 1.056; perp projection: K31 r35 .9873 vs .9870, truth .8794 vs .8767

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_k4diag_v1/k31_coinc_fit.py](code/fable_k4diag_v1/k31_coinc_fit.py)
- [code/fable_k4diag_v1/perp_supply_test.py](code/fable_k4diag_v1/perp_supply_test.py)
- [code/fable_k4diag_v1/planes_covonly_oracle.py](code/fable_k4diag_v1/planes_covonly_oracle.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (12)</summary>

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
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L34](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L34)
- [NOTES.md#L233](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L233)

<details>
<summary>All result files and records (5)</summary>

- [k31_coinc_fit_r35.log](k31_coinc_fit_r35.log) · log
- [perp_supply_test.log](perp_supply_test.log) · log
- [planes_covonly_oracle.log](planes_covonly_oracle.log) · log
- [NOTES.md](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-coincident-index-fourth-cumulant-propagation-adds-too-little-3) · [Copy provenance](../../../PROVENANCE.json)

