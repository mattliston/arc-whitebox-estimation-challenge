# Refit mean encoder on the 40-network validation split: adjusted ratio 0.9848676992236052, 32/40 wins at equal compute

2026-09-21T14:21Z · experiment · adopted · measured

The refit candidate was scored natively against the exact current bank on the 40 reused VALID networks (each side 41 calls in one persistent process, own complete bills, same-history repeat excluded) after three unchanged-export controls matched all 16 output bytes and bills. It passed the prospective gate (pooled adjusted ratio below 1 and at least 24 of 40 wins); mean compute was identical across roles.

Recorded measurement:

    adjusted ratio 0.9848676992236052, raw ratio 0.9848870895876172, adjusted wins 32/40; parent means final_adjusted 3.588077070465302e-09, final_raw 1.9818458171295724e-08, flops 398405465746.925; trained means final_adjusted 3.5337812090261354e-09, final_raw 1.9518943588441375e-08, flops 398405465746.925; 41 new trained calls, 44 reused qualified null/parent calls, trained guard peak 6714179584 bytes

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/run.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/native.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/score_with_lifecycle.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/score_with_lifecycle.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/seal_with_lifecycle.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/seal_with_lifecycle.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)

<details>
<summary>Additional programs and dependencies (27)</summary>

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
- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/endpoint_common.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/endpoint_common.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/fit_common.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/guard.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/lifecycle.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/lifecycle.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/qualify_trained.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/qualify_trained.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/retirement.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/score_native40.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/score_native40.py)
- [code/codex_resume_20260919_v1/current_p_native_retirement_v1/seal_predictions.py](code/codex_resume_20260919_v1/current_p_native_retirement_v1/seal_predictions.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [native40_checkpoint100.md#L3](../../../instruments/native40-scorer-seals-85-calls-requires-lower-own-bill/native40_checkpoint100.md#L3)
- [native40_checkpoint100.md#L9](../../../instruments/native40-scorer-seals-85-calls-requires-lower-own-bill/native40_checkpoint100.md#L9)
- [REEXAMINATION_INDEX_20260919.md#L456](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L456)

<details>
<summary>All result files and records (4)</summary>

- [native40_score.json](native40_score.json) · result
- [native40_checkpoint100.md](../../../instruments/native40-scorer-seals-85-calls-requires-lower-own-bill/native40_checkpoint100.md) · record
- [PROTOCOL.md](../../../instruments/native40-scorer-seals-85-calls-requires-lower-own-bill/PROTOCOL.md) · config
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- ← enables: [Native allocator comparison: peak memory fell from 6,827,851,776 to 2,507,538,432 bytes with exact outputs](../../../records/research/2026-09-21-native-allocator-comparison-peak-memory-fell-6-827-851/README.md)
- measured by: [Native40 scorer seals 85 calls and requires lower own-bill adjusted error with at least 24 of 40 wins](../../../instruments/native40-scorer-seals-85-calls-requires-lower-own-bill/README.md)
- enables: [Submission #331760: 45200 mean-encoder parameters refit on the covariance-edge estimator; score 3.507738856975084e-9](../../../submissions/331760/README.md)
- shipped in: [Submission #331760: 45200 mean-encoder parameters refit on the covariance-edge estimator; score 3.507738856975084e-9](../../../submissions/331760/README.md)
- ← measured by: [Refitting 45200 mean-encoder parameters on the covariance-edge estimator completes 160 graphs and 40 updates](../../learned-correction/refitting-45200-mean-encoder-parameters-covariance-edge-estimator-completes/README.md)

[Research graph](../../../graph/index.html#node=2026-09-21-refit-mean-encoder-40-network-validation-split-adjusted-ratio) · [Copy provenance](../../../PROVENANCE.json)

