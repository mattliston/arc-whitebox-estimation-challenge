# Rank-32 covariance core: audit failed; cross blocks remain necessary

2026-09-06T12:37Z · experiment · informational · measured

September6,2026,05:33:29 PDT. Total actual post13 covariance error on mini57 gregory-wilson, qualified100M public targets; not a network score. Prior-record check CURRENT_COVARIANCE_GEOMETRY_V1.md is important: on old rank35/pre15, top8 retained-core error captured only8.06% ordinary energy, while including both high/complement cross blocks captured81.23%.

Recorded measurement:

    September6,2026,05:33:29 PDT. Total actual post13 covariance error on mini57 gregory-wilson, qualified100M public targets; not a network score. All six input boundary byte checks pass. Stored raw diagonal error .499718 f32ULP, post13->pre14 covariance consistency2.73797e-6 relative, conservative f32 storage bound .027476% of total covariance error RMS. MC uncertainty is not bounded by those storage checks and no noise correction was applied. Input-only top32 risk-tilted bank covariance basis, oracle core coefficients: risk residual .861707, actual next-mean response .823676, ordinary covariance .892811. Oracle top32 tilted-error spectral basis: .328450/.369462/.137720. Both fail the fixed .25 risk AND actual-response gate. Target covariance RMS1.22541108e-4; target next-mean response RMS5.02260501e-5. Fixed-subspace normal stationarity <=7.3e-15. The error spectral basis is a Frobenius truncation comparator, not globally optimal under the trace-augmented risk. Prior-record check CURRENT_COVARIANCE_GEOMETRY_V1.md is important: on old rank35/pre15, top8 retained-core error captured only8.06% ordinary energy, while including both high/complement cross blocks captured81.23%. The present failed retained-core is likewise NOT a test of covariance subspace rotation/tangent correction. A separate fixed-frame cross-block audit is justified, using the already certified score metric and same32 directions; it must not be described as reopening the failed one-high THIRD tensor carriers, changing the frame, or making a correction learnable.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/audit_current_covariance_error_v1.py](code/astra_phase2_reexamination_v1/audit_current_covariance_error_v1.py)
- [code/astra_phase2_reexamination_v1/bound_third_endpoint_score_risk_v1.py](code/astra_phase2_reexamination_v1/bound_third_endpoint_score_risk_v1.py)
- [code/astra_phase2_reexamination_v1/bound_third_endpoint_solve_v1.py](code/astra_phase2_reexamination_v1/bound_third_endpoint_solve_v1.py)
- [code/astra_phase2_reexamination_v1/check_cohort_preplane_gain_v1.py](code/astra_phase2_reexamination_v1/check_cohort_preplane_gain_v1.py)
- [code/astra_phase2_reexamination_v1/check_covariance_score_rank32_oracle_v1.py](code/astra_phase2_reexamination_v1/check_covariance_score_rank32_oracle_v1.py)
- [code/astra_phase2_reexamination_v1/check_covariance_score_tangent_v1.py](code/astra_phase2_reexamination_v1/check_covariance_score_tangent_v1.py)
- [code/astra_phase2_reexamination_v1/check_gaussian_next_mean_risk_metric_v1.py](code/astra_phase2_reexamination_v1/check_gaussian_next_mean_risk_metric_v1.py)
- [code/astra_phase2_reexamination_v1/check_public_p2_third_endpoint_span_v1.py](code/astra_phase2_reexamination_v1/check_public_p2_third_endpoint_span_v1.py)
- [code/astra_phase2_reexamination_v1/check_third_endpoint_score_risk_v1.py](code/astra_phase2_reexamination_v1/check_third_endpoint_score_risk_v1.py)
- [code/astra_phase2_reexamination_v1/check_third_residual_covstar_carry_v1.py](code/astra_phase2_reexamination_v1/check_third_residual_covstar_carry_v1.py)
- [code/astra_phase2_reexamination_v1/cohort_preplane_gain_v1.py](code/astra_phase2_reexamination_v1/cohort_preplane_gain_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)

<details>
<summary>Additional programs and dependencies (26)</summary>

- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/covariance_score_subspace_v1.py](code/astra_phase2_reexamination_v1/covariance_score_subspace_v1.py)
- [code/astra_phase2_reexamination_v1/covariance_score_tangent_v1.py](code/astra_phase2_reexamination_v1/covariance_score_tangent_v1.py)
- [code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py](code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_plane_pair_span_v1.py](code/astra_phase2_reexamination_v1/fourth_plane_pair_span_v1.py)
- [code/astra_phase2_reexamination_v1/freeze_cohort_preplane_gain_v1.py](code/astra_phase2_reexamination_v1/freeze_cohort_preplane_gain_v1.py)
- [code/astra_phase2_reexamination_v1/freeze_public_p2_third_carry_inputs_v1.py](code/astra_phase2_reexamination_v1/freeze_public_p2_third_carry_inputs_v1.py)
- [code/astra_phase2_reexamination_v1/freeze_public_p2_third_cell_v1.py](code/astra_phase2_reexamination_v1/freeze_public_p2_third_cell_v1.py)
- [code/astra_phase2_reexamination_v1/freeze_public_p2_third_next_response_v1.py](code/astra_phase2_reexamination_v1/freeze_public_p2_third_next_response_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_next_mean_risk_metric_v1.py](code/astra_phase2_reexamination_v1/gaussian_next_mean_risk_metric_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/public_third_centering_v1.py](code/astra_phase2_reexamination_v1/public_third_centering_v1.py)
- [code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py](code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py)
- [code/astra_phase2_reexamination_v1/qualify_public_p2_first_moment_v1.py](code/astra_phase2_reexamination_v1/qualify_public_p2_first_moment_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/third_endpoint_score_risk_v1.py](code/astra_phase2_reexamination_v1/third_endpoint_score_risk_v1.py)
- [code/astra_phase2_reexamination_v1/third_endpoint_span_v1.py](code/astra_phase2_reexamination_v1/third_endpoint_span_v1.py)
- [code/astra_phase2_reexamination_v1/third_residual_covstar_v1.py](code/astra_phase2_reexamination_v1/third_residual_covstar_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/data_gate/target_blind_mini_v2.py](code/data_gate/target_blind_mini_v2.py)

</details>

## Evidence

- [COVARIANCE_SCORE_RANK32_ORACLE_V1_VERDICT.md#L1](COVARIANCE_SCORE_RANK32_ORACLE_V1_VERDICT.md#L1)
- [COVARIANCE_SCORE_RANK32_ORACLE_V1_VERDICT.md#L15](COVARIANCE_SCORE_RANK32_ORACLE_V1_VERDICT.md#L15)

<details>
<summary>All result files and records (4)</summary>

- [COVARIANCE_SCORE_RANK32_ORACLE_V1_VERDICT.md](COVARIANCE_SCORE_RANK32_ORACLE_V1_VERDICT.md) · record
- [covariance_score_rank32_oracle_v1.json](covariance_score_rank32_oracle_v1.json) · result
- [CURRENT_COVARIANCE_GEOMETRY_V1.md](CURRENT_COVARIANCE_GEOMETRY_V1.md) · record
- [COVARIANCE_SCORE_RANK32_ORACLE_DESIGN_V1.md](COVARIANCE_SCORE_RANK32_ORACLE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-rank-32-covariance-core-audit-failed-cross-blocks-remain) · [Copy provenance](../../../PROVENANCE.json)

