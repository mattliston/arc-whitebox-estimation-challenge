# Independent complete-pair thinning fails the final-consumer screen

2026-09-06T17:42Z · experiment · rejected · measured

Both miss the declared1e-9 allowance by about8x. These are expected LOCAL perturbation variances at the frozen bank's actual final readout, not raw errors against truth or adjusted scores. Additional archive check after this result: omitting CP output rows near the ReLU linear tails is NOT novel.

Recorded measurement:

    | Net | Original pairs | Expected retained | Optimal added MSE | Uniform added MSE | |---|---:|---:|---:|---:| |35|11936|5968|7.97996804392e-9|8.82063052238e-8| |100|11872|5936|7.94569647108e-9|7.92275660941e-8| Both miss the declared1e-9 allowance by about8x. These are expected LOCAL perturbation variances at the frozen bank's actual final readout, not raw errors against truth or adjusted scores. Earlier-layer error, state feedback, probability acquisition and all runtime costs were free. Close this independent-pair proposal: no budget/probability sweep, draws, whole carrier, coadaptation, panel or upload. It is not a universal bound on deterministic or correlated compression, or a coadapted estimator. The numerical prerequisites pass. Full MPS baseline max-mean errors 1.39839787394e-6 /1.39716386938e-6 are below the unchanged2e-6 gate; all30 active/birth memberships match. Complete-pair summation agrees with the original signed cubes to5.61e-16 /5.25e-16 relative. The exact diagonal-plus-rank-one final Jacobian agrees with independent reverse derivatives to2.34e-8 /2.05e-8. Tiny exhaustive Bernoulli enumeration matches the variance formula exactly at reported precision. One nice10 worker,3 CPU threads, one MPS job;3.381seconds. Peak process RSS2,062,761,984B; largest observed GPU driver allocation6,478,872,576B. These are not independent additive peaks. No threadpoolctl or MPS fallback. No target labels loaded. Additional archive check after this result: omitting CP output rows near the ReLU linear tails is NOT novel. RESPONSE_ROWS_MATCHED_ROUND_V1_CLOSED.md already records its exact-linear certificate,200-row matched fit and failed3% raw-worsening gate on rank35. Do not rebrand that proposal as a new saturation-aware read strategy or repeat its threshold experiment.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py](code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py)
- [code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py](code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py)
- [code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py](code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_d5_t8192_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_pair_polynomial_f32_v1.py](code/astra_phase2_reexamination_v1/candidate_response_pair_polynomial_f32_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py](code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_whole_pair_kernel_post7_k512_v1.py](code/astra_phase2_reexamination_v1/candidate_whole_pair_kernel_post7_k512_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)

<details>
<summary>Additional programs and dependencies (31)</summary>

- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/chain_radial_training.py](code/astra_phase2_reexamination_v1/chain_radial_training.py)
- [code/astra_phase2_reexamination_v1/chain_response_rows_training_v1.py](code/astra_phase2_reexamination_v1/chain_response_rows_training_v1.py)
- [code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py](code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/generate_radial_rows.py](code/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/astra_phase2_reexamination_v1/generate_response_rows_v1.py](code/astra_phase2_reexamination_v1/generate_response_rows_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/paired_cp_thinning_bound_v1.py](code/astra_phase2_reexamination_v1/paired_cp_thinning_bound_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/radial_response_rows_v1.py](code/astra_phase2_reexamination_v1/radial_response_rows_v1.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/capture_candidate.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/capture_candidate.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/common.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/guard.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/seal.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/seal.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [PAIRED_CP_THINNING_BOUND_V1_CLOSED.md#L1](PAIRED_CP_THINNING_BOUND_V1_CLOSED.md#L1)
- [PAIRED_CP_THINNING_BOUND_V1_CLOSED.md#L12](PAIRED_CP_THINNING_BOUND_V1_CLOSED.md#L12)
- [PAIRED_CP_THINNING_BOUND_V1_CLOSED.md#L23](PAIRED_CP_THINNING_BOUND_V1_CLOSED.md#L23)

<details>
<summary>All result files and records (3)</summary>

- [PAIRED_CP_THINNING_BOUND_V1_CLOSED.md](PAIRED_CP_THINNING_BOUND_V1_CLOSED.md) · record
- [PAIRED_CP_THINNING_BOUND_DESIGN_V1.md](PAIRED_CP_THINNING_BOUND_DESIGN_V1.md) · record
- [RESPONSE_ROWS_MATCHED_ROUND_V1_CLOSED.md](../../flop-accounting/response-row-matched-round-closed-declared-gate/RESPONSE_ROWS_MATCHED_ROUND_V1_CLOSED.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-independent-complete-pair-thinning-fails-final-consumer-screen) · [Copy provenance](../../../PROVENANCE.json)

