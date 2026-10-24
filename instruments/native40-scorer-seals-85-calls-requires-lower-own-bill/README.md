# Native40 scorer seals 85 calls and requires lower own-bill adjusted error with at least 24 of 40 wins

2026-09-21 · tool · informational · measured

The endpoint scorer authenticates unchanged-export controls, a persistent reference sequence and the trained sequence before decoding validation targets. Its memory-retirement successor reuses already-qualified reference calls and preserves the original roster, scoring and consistency gates.

Recorded measurement:

    85sealed reused/new prediction and lifecycle records;44qualified calls reused,41new trained calls;all16 domains,45selectors;7.5GB guard;120s prediction limit;adjusted ratio<1 and at least24/40 wins.

## Programs

- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/endpoint_common.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/endpoint_common.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/fit_common.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/guard.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/lifecycle.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/lifecycle.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/native.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/prepare.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/prepare.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/qualify_trained.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/qualify_trained.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/retirement.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/run.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/score_native40.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/score_native40.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/score_with_lifecycle.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/score_with_lifecycle.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/seal_predictions.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/seal_predictions.py)

<details>
<summary>Additional programs and dependencies (42)</summary>

- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/seal_with_lifecycle.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/seal_with_lifecycle.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/alloc_common.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/alloc_common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/guard.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/guard.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/candidate.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py)
- [code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/endpoint_common.py](code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/endpoint_common.py)
- [code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/fit_common.py](code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/lifecycle.py](code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/lifecycle.py)
- [code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/native.py](code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/retirement.py](code/codex_resume_20260919_v1/current_p_native_allocator_qualification_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/endpoint_common.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/endpoint_common.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/fit_common.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/lifecycle.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/lifecycle.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/native.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/seal_predictions.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/seal_predictions.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/seal_with_lifecycle.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/seal_with_lifecycle.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [PROTOCOL.md#L40](PROTOCOL.md#L40)
- [native40_checkpoint100.md#L3](native40_checkpoint100.md#L3)

<details>
<summary>All result files and records (2)</summary>

- [PROTOCOL.md](PROTOCOL.md) · record
- [native40_checkpoint100.md](native40_checkpoint100.md) · record

</details>

## Related work

- ← measured by: [Refit mean encoder on the 40-network validation split: adjusted ratio 0.9848676992236052, 32/40 wins at equal compute](../../experiments/evaluation-methodology/refit-mean-encoder-40-network-validation-split-adjusted-ratio/README.md)

[Research graph](../../graph/index.html#node=2026-09-21-native40-scorer-seals-85-calls-requires-lower-own-bill) · [Copy provenance](../../PROVENANCE.json)

