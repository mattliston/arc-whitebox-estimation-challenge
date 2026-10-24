# Residual covariance/coordinate-star carry: closed

2026-09-06T12:09Z · experiment · rejected · measured

Dense small tensor/affine certificates passed<=1.11e-16. All18 original pre13/pre14 input and prefix checks were byte-exact; pre15 response comes from the actual scalar-final path. Natural oracle carry leaves covariance-response error ratio .8802095124 and next-mean response ratio .8686113339 (both required<=.75).

Recorded measurement:

    Dense small tensor/affine certificates passed<=1.11e-16. All18 original pre13/pre14 input and prefix checks were byte-exact; pre15 response comes from the actual scalar-final path. Source repeated-plane relative error 1.25e-16. New pre14 target checks pass: affine mean1.01e-10 RMS, covariance 2.74e-6 relative, stored-f32 rounding bound .0003144 of missing-plane RMS. No clipping, fit, new samples or additional public file. Natural oracle carry leaves covariance-response error ratio .8802095124 and next-mean response ratio .8686113339 (both required<=.75). Cosines .35736/.36262. Weighted K21 error ratio .8371206130. Source error norm 1.71942; propagated scalar marginal was excluded as designed. FAIL.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_third_residual_covstar_carry_v1.py](code/astra_phase2_reexamination_v1/check_third_residual_covstar_carry_v1.py)
- [code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py](code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_plane_pair_span_v1.py](code/astra_phase2_reexamination_v1/fourth_plane_pair_span_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/public_third_centering_v1.py](code/astra_phase2_reexamination_v1/public_third_centering_v1.py)
- [code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py](code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/third_endpoint_span_v1.py](code/astra_phase2_reexamination_v1/third_endpoint_span_v1.py)
- [code/astra_phase2_reexamination_v1/third_residual_covstar_v1.py](code/astra_phase2_reexamination_v1/third_residual_covstar_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

## Evidence

- [THIRD_RESIDUAL_COVSTAR_CARRY_V1_CLOSED.md#L1](THIRD_RESIDUAL_COVSTAR_CARRY_V1_CLOSED.md#L1)
- [THIRD_RESIDUAL_COVSTAR_CARRY_V1_CLOSED.md#L11](THIRD_RESIDUAL_COVSTAR_CARRY_V1_CLOSED.md#L11)
- [THIRD_RESIDUAL_COVSTAR_CARRY_V1_CLOSED.md#L17](THIRD_RESIDUAL_COVSTAR_CARRY_V1_CLOSED.md#L17)

<details>
<summary>All result files and records (3)</summary>

- [THIRD_RESIDUAL_COVSTAR_CARRY_V1_CLOSED.md](THIRD_RESIDUAL_COVSTAR_CARRY_V1_CLOSED.md) · record
- [third_residual_covstar_carry_v1.json](third_residual_covstar_carry_v1.json) · result
- [THIRD_RESIDUAL_COVSTAR_CARRY_DESIGN_V1.md](THIRD_RESIDUAL_COVSTAR_CARRY_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-residual-covariance-coordinate-star-carry-closed) · [Copy provenance](../../../PROVENANCE.json)

