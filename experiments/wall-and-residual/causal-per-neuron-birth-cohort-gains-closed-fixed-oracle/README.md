# Causal per-neuron birth-cohort gains: closed at fixed oracle gate

2026-09-06T13:06Z · experiment · rejected · measured

Small signed-factor derivative relative error3.13e-11; exact finite action 1.06e-16 absolute; pair consumer2.78e-17; dense solve relative3.81e-12. Wide104-feature SVD retained full rank, reconstruction2.00e-15. Both .50 response gates fail.

Recorded measurement:

    Small signed-factor derivative relative error3.13e-11; exact finite action 1.06e-16 absolute; pair consumer2.78e-17; dense solve relative3.81e-12. Wide104-feature SVD retained full rank, reconstruction2.00e-15. Sum of f64 cohort planes versus parent's f32 plane differs by only2.56706e-6 of the true residual norm, far below the1% numerical admission gate. All six current next-response geometry arrays reproduce exactly. Fixed .001-ridge1352-parameter score-risk fit converged in92 iterations, true normal residual8.61829e-8 <=1e-7. Results relative to the complete isolated off-diagonal true-K21 covariance and following-mean responses: Fit wall2.70462s, peak RSS1,523,613,696 bytes. Acquisition wall5.24204s, peak RSS4,193,730,560 bytes. Receipt cohort_preplane_gain_wide_v1.json; output SHA256 a211e0583706d79bd59af8771ae67459671188424706823eb29515d8a73cc605.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_cohort_preplane_gain_v1.py](code/astra_phase2_reexamination_v1/check_cohort_preplane_gain_v1.py)
- [code/astra_phase2_reexamination_v1/cohort_preplane_gain_v1.py](code/astra_phase2_reexamination_v1/cohort_preplane_gain_v1.py)
- [code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py](code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_plane_pair_span_v1.py](code/astra_phase2_reexamination_v1/fourth_plane_pair_span_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_next_mean_risk_metric_v1.py](code/astra_phase2_reexamination_v1/gaussian_next_mean_risk_metric_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py](code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/third_endpoint_score_risk_v1.py](code/astra_phase2_reexamination_v1/third_endpoint_score_risk_v1.py)
- [code/astra_phase2_reexamination_v1/third_endpoint_span_v1.py](code/astra_phase2_reexamination_v1/third_endpoint_span_v1.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [COHORT_PREPLANE_GAIN_V1_CLOSED.md#L1](COHORT_PREPLANE_GAIN_V1_CLOSED.md#L1)
- [COHORT_PREPLANE_GAIN_V1_CLOSED.md#L15](COHORT_PREPLANE_GAIN_V1_CLOSED.md#L15)

<details>
<summary>All result files and records (3)</summary>

- [COHORT_PREPLANE_GAIN_V1_CLOSED.md](COHORT_PREPLANE_GAIN_V1_CLOSED.md) · record
- [cohort_preplane_gain_wide_v1.json](cohort_preplane_gain_wide_v1.json) · result
- [COHORT_PREPLANE_GAIN_DESIGN_V1.md](COHORT_PREPLANE_GAIN_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-causal-per-neuron-birth-cohort-gains-closed-fixed-oracle) · [Copy provenance](../../../PROVENANCE.json)

