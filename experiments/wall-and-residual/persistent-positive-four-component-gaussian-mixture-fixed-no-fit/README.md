# Persistent positive four-component Gaussian mixture: fixed no-fit raw test failed

2026-09-06T22:10Z · experiment · rejected · measured

The fixed requirements were BOTH raw<=6e-8 AND ratio<=.25 on EACH network. Neither is close. The frozen no-fit component/source/direction/power/map/merge round is closed.

Recorded measurement:

    | Network rank | Single-Gaussian control | GMM4 | GMM4 / control | | --- | ---: | ---: | ---: | | 35 | 2.89617838895e-6 | 2.61823343950e-6 | .9040304456 | | 100 | 4.63004606124e-6 | 4.25983219808e-6 | .9200409978 | The fixed requirements were BOTH raw<=6e-8 AND ratio<=.25 on EACH network. Neither is close. The observed8–10% improvement over an untrained Gaussian control is not competitive with the existing learned full-pair bank. These are raw MSEs, not adjusted scores, panel means or official submission scores. Do not multiply by the provisional cost envelope and advertise a validated adjusted candidate. No new Monte Carlo or training was performed. - Rank35 single/GMM wall: .7533/4.0350s; rank100: .8307/3.9105s. Whole smoke including scoring took9.6163s. These are offline NumPy timings, not grader persistent wall or residual-time gates. - Maximum process RSS1,203,224,576 bytes, below2GiB. One nice10 NumPy-only worker; requested two BLAS threads via environment and threadpool_limits. Empty backend instrumentation does not independently prove actual count. - Each GMM used115 full FP32 products,59 FP64 Choleskys,86 pair maps, 335 merge-pair comparisons and280 direction actions, exactly the design. - All Choleskys succeeded on first attempt. No jitter, eigensolver or retry. Mean/variance roundoff lifts were zero; largest diagonal-tail roundoff lift 8.3267e-17. Largest factor-rounding relative error2.6066e-8; no positive variance coordinate was zeroed by factor underflow. - Retained L32 factors own PSD mixture state. Rounded affine/child matrices remain approximate numerical inputs, not separately PSD-certified objects. - Full provisional charge324,194,464,597.33 FLOPs, utilization .147426574, includes a35B auxiliary reserve. This remains an unmetered cost envelope. The tiny native Cholesky operation check does not qualify a whole estimator. The frozen no-fit component/source/direction/power/map/merge round is closed. This is not a general impossibility theorem for learned mixtures or models with a non-Gaussian transverse state. Such a state would need genuinely new source information and full cost accounting, not free inherited CP trajectories. Official bank #329926 remains4.982231460183304e-9 adjusted (not freshly polled); the <1e-9 goal remains active and unmet. No threshold bank was earned here.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py](code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py)
- [code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py](code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py)
- [code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py](code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_d5_t8192_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py](code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py)
- [code/astra_phase2_reexamination_v1/certify_gmm_gaussian_query_merge_v1.py](code/astra_phase2_reexamination_v1/certify_gmm_gaussian_query_merge_v1.py)
- [code/astra_phase2_reexamination_v1/certify_positive_gmm_merge_algebra_v1.py](code/astra_phase2_reexamination_v1/certify_positive_gmm_merge_algebra_v1.py)
- [code/astra_phase2_reexamination_v1/check_cumulant_tilt_response_v1.py](code/astra_phase2_reexamination_v1/check_cumulant_tilt_response_v1.py)

<details>
<summary>Additional programs and dependencies (23)</summary>

- [code/astra_phase2_reexamination_v1/check_cycle_direct_mean_witness_v1.py](code/astra_phase2_reexamination_v1/check_cycle_direct_mean_witness_v1.py)
- [code/astra_phase2_reexamination_v1/check_gaussian_fourth_r1_source_v1.py](code/astra_phase2_reexamination_v1/check_gaussian_fourth_r1_source_v1.py)
- [code/astra_phase2_reexamination_v1/check_positive_path_value_transfer_v1.py](code/astra_phase2_reexamination_v1/check_positive_path_value_transfer_v1.py)
- [code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py](code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py)
- [code/astra_phase2_reexamination_v1/cumulant_tilt_response_v1.py](code/astra_phase2_reexamination_v1/cumulant_tilt_response_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/linear_transport_residual_source_v1.py](code/astra_phase2_reexamination_v1/linear_transport_residual_source_v1.py)
- [code/astra_phase2_reexamination_v1/positive_gmm4_split_merge_prototype_v1.py](code/astra_phase2_reexamination_v1/positive_gmm4_split_merge_prototype_v1.py)
- [code/astra_phase2_reexamination_v1/positive_path_birth_reference_v1.py](code/astra_phase2_reexamination_v1/positive_path_birth_reference_v1.py)
- [code/astra_phase2_reexamination_v1/positive_path_resummation_v1.py](code/astra_phase2_reexamination_v1/positive_path_resummation_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/radial_response_rows_v1.py](code/astra_phase2_reexamination_v1/radial_response_rows_v1.py)
- [code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py](code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/capture_candidate.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/capture_candidate.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/common.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/guard.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/seal.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/seal.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [POSITIVE_GMM4_SPLIT_MERGE_V1_CLOSED.md#L1](POSITIVE_GMM4_SPLIT_MERGE_V1_CLOSED.md#L1)
- [POSITIVE_GMM4_SPLIT_MERGE_V1_CLOSED.md#L23](POSITIVE_GMM4_SPLIT_MERGE_V1_CLOSED.md#L23)
- [POSITIVE_GMM4_SPLIT_MERGE_V1_CLOSED.md#L34](POSITIVE_GMM4_SPLIT_MERGE_V1_CLOSED.md#L34)

<details>
<summary>All result files and records (3)</summary>

- [POSITIVE_GMM4_SPLIT_MERGE_V1_CLOSED.md](POSITIVE_GMM4_SPLIT_MERGE_V1_CLOSED.md) · record
- [result.json](result.json) · result
- [POSITIVE_GMM4_SPLIT_MERGE_DESIGN_V1.md](POSITIVE_GMM4_SPLIT_MERGE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-persistent-positive-four-component-gaussian-mixture-fixed-no-fit) · [Copy provenance](../../../PROVENANCE.json)

