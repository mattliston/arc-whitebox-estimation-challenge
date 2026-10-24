# Public third-plane endpoint span: fixed objective closed

2026-09-06T11:51Z · experiment · rejected · measured

The newly discovered public100M joint-moment dataset is numerically qualified on the preselected first training mini network57, gregory-wilson. Its pinned337MB file passed SHA256 and identity checks. No extra iterations followed.

Recorded measurement:

    The newly discovered public100M joint-moment dataset is numerically qualified on the preselected first training mini network57, gregory-wilson. Its pinned337MB file passed SHA256 and identity checks. Source seed/sample count match the contributor card. First Gaussian-layer standardized RMS errors are1.00116 for covariance and1.00025 for K21. Pre13 affine mean discrepancy8.96e-11 RMS, covariance2.66e-6 relative. Stored-f32 K21 rounding bound is .00028057 of on-policy missing-plane RMS. This is not a bound on finite-sample noise or shared GPU bias. The 100M post-mean array was decoded for centering/identity; official1e9 gt_mean was NOT decoded, and no final score was calculated. No held-out moments were opened. Before labels, actual bank prefix and current64-state features froze. Independent pre14-response replay reproduced all15 input/prefix arrays byte-for-byte. Weighted adjoint and small dense ridge solution certified; wide third-to-covariance kernel agrees with exact substitution to2.17e-19. F=[1,H64] has rank65, span reconstruction1.48e-15. Three rho powers, unrestricted65x65 coefficients, weighted K21 loss, ridge.001 were fixed. The300-iteration solve did NOT pass tolerance: normal residual2.158887e-5 versus1e-7. Its in-sample isolated-response error ratios were: No extra iterations followed. A separately certified posterior bound uses A >= .001 I and the exact quadratic-response Gram row-sum norm bound. With1.001 safety inflation, the UNIQUE EXACT optimum of this same ridge objective still has next-mean error ratio >=.529467652, above the unchanged .50 gate. Thus that value gate cannot be rescued just by finishing this solve. The initial bound script failed only on JSON NumPy-bool serialization; v2 preserved it and repaired serialization, with zero refitting iterations. This closes the fixed endpoint dictionary/objective test, not all third predictors or an unregularized universal representation. Single-network in-sample oracle coefficients are not cross-network learned predictions, total covariance errors, final raw MSE or adjusted scores. No remaining seven-file download,6/2 fit, parameter/objective/feature sweep, state write, port, panel or upload follows under this design. Public source and attribution: https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026 commit1bb00b739619154ca255db6443af0b6a9dd960ce, contributor MIT card; original WhestBench v2-phase2 CC-BY-4.0. Source file SHA256 f2c79442c4adbf4e12a7db98f6a72a74c66d9c50872f99f8ff63599834d72670.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/audit_public_p2_moments_metadata_v1.py](code/astra_phase2_reexamination_v1/audit_public_p2_moments_metadata_v1.py)
- [code/astra_phase2_reexamination_v1/bound_third_endpoint_solve_v1.py](code/astra_phase2_reexamination_v1/bound_third_endpoint_solve_v1.py)
- [code/astra_phase2_reexamination_v1/bound_third_endpoint_solve_v2.py](code/astra_phase2_reexamination_v1/bound_third_endpoint_solve_v2.py)
- [code/astra_phase2_reexamination_v1/check_public_p2_third_endpoint_span_v1.py](code/astra_phase2_reexamination_v1/check_public_p2_third_endpoint_span_v1.py)
- [code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py](code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py)
- [code/astra_phase2_reexamination_v1/download_public_p2_first_moment_v1.py](code/astra_phase2_reexamination_v1/download_public_p2_first_moment_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_plane_pair_span_v1.py](code/astra_phase2_reexamination_v1/fourth_plane_pair_span_v1.py)
- [code/astra_phase2_reexamination_v1/freeze_public_p2_third_cell_v1.py](code/astra_phase2_reexamination_v1/freeze_public_p2_third_cell_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/public_third_centering_v1.py](code/astra_phase2_reexamination_v1/public_third_centering_v1.py)
- [code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py](code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py)
- [code/astra_phase2_reexamination_v1/qualify_public_p2_first_moment_v1.py](code/astra_phase2_reexamination_v1/qualify_public_p2_first_moment_v1.py)

<details>
<summary>Additional programs and dependencies (10)</summary>

- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/third_endpoint_span_v1.py](code/astra_phase2_reexamination_v1/third_endpoint_span_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/data_gate/target_blind_mini_v2.py](code/data_gate/target_blind_mini_v2.py)

</details>

## Evidence

- [PUBLIC_P2_THIRD_ENDPOINT_SPAN_V1_CLOSED.md#L1](PUBLIC_P2_THIRD_ENDPOINT_SPAN_V1_CLOSED.md#L1)

<details>
<summary>All result files and records (6)</summary>

- [PUBLIC_P2_THIRD_ENDPOINT_SPAN_V1_CLOSED.md](PUBLIC_P2_THIRD_ENDPOINT_SPAN_V1_CLOSED.md) · record
- [public_p2_moments_metadata_v1.json](public_p2_moments_metadata_v1.json) · result
- [public_p2_first_moment_qualification_v1.json](public_p2_first_moment_qualification_v1.json) · result
- [public_p2_third_endpoint_span_v1.json](public_p2_third_endpoint_span_v1.json) · result
- [third_endpoint_solve_bound_v2.json](third_endpoint_solve_bound_v2.json) · result
- [PUBLIC_P2_THIRD_ENDPOINT_SPAN_DESIGN_V1.md](PUBLIC_P2_THIRD_ENDPOINT_SPAN_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-public-third-plane-endpoint-span-fixed-objective-closed) · [Copy provenance](../../../PROVENANCE.json)

