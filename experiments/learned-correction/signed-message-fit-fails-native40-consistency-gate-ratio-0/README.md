# Signed-message fit fails native40 consistency gate: ratio 0.999179297214, 23/40 wins versus 24 required

2026-09-21 · experiment · rejected · measured

The complete 160-graph and 40-update fit was evaluated with each native call's own metered cost.

Recorded measurement:

    The complete 160-graph/40-update fit and 86 native calls finished. The actual
    40-network own-cost adjusted comparison is 0.999179297214, with 23/40 wins.
    It misses the original 24-win gate. Raw ratio is 0.998683256394; the
    0.082% adjusted improvement is too small to justify promotion. No downstream
    be restarted. The old session84084 handle is missing and current process
    inspection also found no remaining Phase2 worker.
    All 669 prediction-seal pins were rehashed, both means and the gate were
    recomputed from all40 scalar rows, and all86 call-retirement receipts and
    8 stage guards passed. Peak sampled RSS was
    7.370GB; added swap was zero. This confirms
    Exact official331760 remains3.507738856975084e-9,100/100,zero failures.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/build.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/build.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/candidate.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/common.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/endpoint_common.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/endpoint_common.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/guard.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/native.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/null_candidate.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/null_candidate.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/prepare.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/prepare.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/qualify_null.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/qualify_null.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/records.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/records.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/retirement.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/run.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/run.py)

<details>
<summary>Additional programs and dependencies (42)</summary>

- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/score_native40.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/score_native40.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/seal_predictions.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/seal_predictions.py)
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
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/endpoint_common.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/endpoint_common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/fit_common.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/guard.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/score_native40.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/score_native40.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/guard.py](code/codex_resume_20260919_v1/current_p_batch_process_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/common.py](code/codex_resume_20260919_v1/current_p_release_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/native.py](code/codex_resume_20260919_v1/current_p_release_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_endpoint_v1/build.py](code/codex_resume_20260919_v1/current_p_signed_branch_endpoint_v1/build.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_endpoint_v1/common.py](code/codex_resume_20260919_v1/current_p_signed_branch_endpoint_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_endpoint_v1/endpoint_common.py](code/codex_resume_20260919_v1/current_p_signed_branch_endpoint_v1/endpoint_common.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_endpoint_v1/guard.py](code/codex_resume_20260919_v1/current_p_signed_branch_endpoint_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L6](RESULT.md#L6)
- [REEXAMINATION_INDEX_20260919.md#L388](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L388)
- [REEXAMINATION_INDEX_20260919.md#L394](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L394)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-signed-message-fit-fails-native40-consistency-gate-ratio-0) · [Copy provenance](../../../PROVENANCE.json)

