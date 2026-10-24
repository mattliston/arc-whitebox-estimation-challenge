# Score-risk endpoint fit: closed

2026-09-06T12:31Z · experiment · rejected · measured

- Analytic mean-response risk residual .772535 (old fit .828883). - Actual W14 mean-response residual .801489 (old fit .859562). - Ordinary covariance residual .460450, old weighted-plane residual1.22507. - Exactly zero marginal mean write; pair-kernel parity1.63e-19. - Fixed1000-step solve true relative residual4.20655e-7, missing1e-7 gate. No extra iterations were run. September 6, 2026.

Recorded measurement:

    - Analytic mean-response risk residual .772535 (old fit .828883). - Actual W14 mean-response residual .801489 (old fit .859562). - Ordinary covariance residual .460450, old weighted-plane residual1.22507. - Exactly zero marginal mean write; pair-kernel parity1.63e-19. - Fixed1000-step solve true relative residual4.20655e-7, missing1e-7 gate. No extra iterations were run. The one-normal-application, same-ridge spectral displacement bound gives exact-optimum risk ratio >=.77252535, far above the required .50. This bound is algebraic with a1.001 numerical margin, not interval arithmetic, a statistical certificate, an unregularized capacity bound, or a statement about all possible predictors. The fit took20.61s, peak host RSS1,145,618,432 bytes. Fit finished05:22:42 PDT and bound05:28:12 PDT, September6,2026. No new labels or files downloaded, no held-out access, carrier write, raw/adjusted score, or estimator change. No regularizer, endpoint rank, metric or solver-budget continuation.

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

- [THIRD_ENDPOINT_SCORE_RISK_V1_CLOSED.md#L1](THIRD_ENDPOINT_SCORE_RISK_V1_CLOSED.md#L1)

<details>
<summary>All result files and records (5)</summary>

- [THIRD_ENDPOINT_SCORE_RISK_V1_CLOSED.md](THIRD_ENDPOINT_SCORE_RISK_V1_CLOSED.md) · record
- [third_endpoint_score_risk_wide_v1.json](third_endpoint_score_risk_wide_v1.json) · result
- [third_endpoint_score_risk_bound_v1.json](third_endpoint_score_risk_bound_v1.json) · result
- [THIRD_ENDPOINT_SCORE_RISK_DESIGN_V1.md](THIRD_ENDPOINT_SCORE_RISK_DESIGN_V1.md) · record
- [THIRD_ENDPOINT_SCORE_RISK_BOUND_DESIGN_V1.md](THIRD_ENDPOINT_SCORE_RISK_BOUND_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-score-risk-endpoint-fit-closed) · [Copy provenance](../../../PROVENANCE.json)

