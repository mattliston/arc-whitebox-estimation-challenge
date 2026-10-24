# Frozen-variance Gaussian risk audit: passed

2026-09-06T12:21Z · experiment · informational · measured

All six actual pre14 boundary arrays reproduce byte-exactly. Raw affine mean was not stored; alpha, sigma, and Gaussian mean reproduce exactly, and the alpha*sigma round-trip differs by at most 4.44e-16. All predeclared gates passed.

Recorded measurement:

    All six actual pre14 boundary arrays reproduce byte-exactly. Raw affine mean was not stored; alpha, sigma, and Gaussian mean reproduce exactly, and the alpha*sigma round-trip differs by at most 4.44e-16. Small explicit Gaussian-fourth-moment certification has relative error <=3.58e-16. | Field | Analytic risk | Varying-variance Monte Carlo / analytic | |---|---:|---:| | Target covariance response | 4.3616661e-10 | 1.04629 | | Fixed predicted response | 9.8543206e-11 | 1.03800 | | Residual | 3.6153127e-10 | 1.05485 | Frozen-variance analytic/MC differences are <=0.752 batch standard errors. Analytic residual ratio .828883 versus varying-variance MC .835662 (batch SE .020874): relative disagreement .008112, passing the fixed .15 gate. Actual W14 ratio .859562, compared with ordinary covariance Frobenius ratio .468996. Mean-ray suppression beta=.952658. This confirms a substantial loss mismatch for this fixed field and training cell, not that changing the loss necessarily creates a learnable correction.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

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
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)

<details>
<summary>Additional programs and dependencies (25)</summary>

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

- [GAUSSIAN_NEXT_MEAN_RISK_METRIC_V1_VERDICT.md#L1](GAUSSIAN_NEXT_MEAN_RISK_METRIC_V1_VERDICT.md#L1)
- [GAUSSIAN_NEXT_MEAN_RISK_METRIC_V1_VERDICT.md#L10](GAUSSIAN_NEXT_MEAN_RISK_METRIC_V1_VERDICT.md#L10)
- [GAUSSIAN_NEXT_MEAN_RISK_METRIC_V1_VERDICT.md#L18](GAUSSIAN_NEXT_MEAN_RISK_METRIC_V1_VERDICT.md#L18)
- [GAUSSIAN_NEXT_MEAN_RISK_METRIC_V1_VERDICT.md#L20](GAUSSIAN_NEXT_MEAN_RISK_METRIC_V1_VERDICT.md#L20)

<details>
<summary>All result files and records (3)</summary>

- [GAUSSIAN_NEXT_MEAN_RISK_METRIC_V1_VERDICT.md](GAUSSIAN_NEXT_MEAN_RISK_METRIC_V1_VERDICT.md) · record
- [gaussian_next_mean_risk_metric_wide_v1.json](gaussian_next_mean_risk_metric_wide_v1.json) · result
- [GAUSSIAN_NEXT_MEAN_RISK_METRIC_DESIGN_V1.md](GAUSSIAN_NEXT_MEAN_RISK_METRIC_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-frozen-variance-gaussian-risk-audit-passed) · [Copy provenance](../../../PROVENANCE.json)

