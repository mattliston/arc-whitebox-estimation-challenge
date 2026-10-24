# Centered moment rewrite: scalar, pair and full-depth float64 fidelity passed

2026-09-06T13:38Z · experiment · informational · measured

Scalar66-cell120-digit certificate (06:29:10PDT): standardized means [-12,-8,-4,-2,-.5,0,.5,2,4,8,12], sigmas .3/1.7, three fixed(k3,k4) cases. Maximum sigma-normalized absolute errors: mean2.115e-15, variance1.880e-16, third cumulant2.410e-16, fourth cumulant7.277e-16. Next work is a versioned Torch port and explicit f32 numerical/gradient validation, not training.

Recorded measurement:

    Scalar66-cell120-digit certificate (06:29:10PDT): standardized means [-12,-8,-4,-2,-.5,0,.5,2,4,8,12], sigmas .3/1.7, three fixed(k3,k4) cases. Maximum sigma-normalized absolute errors: mean2.115e-15, variance1.880e-16, third cumulant2.410e-16, fourth cumulant7.277e-16. Uses reflected negative Gaussian tail central moments and exact centering of Edgeworth raw increments, including all powers of the small mean shift. Pair-map certificate (06:32:41PDT) passes for widths11/17, symmetric and deliberately1e-6-asymmetric covariance. Maximum normalized K21 error3.09e-14, radial error9.08e-15. Crucial asymmetry term is retained: the old map uses p21.T, which is NOT exactly pair(A,B) when f32 Cpre is asymmetric. Direct small antisymmetric powers and K22 weights reproduce that difference; the bank covariance is not silently symmetrized. | Development net | Max mean difference | Final mean difference RMS | Final head-delta max | |---|---:|---:|---:| |35|1.81815089e-7|3.20575814e-8|8.25821189e-10| |100|1.86008903e-7|3.60893611e-8|1.24407507e-9| All30 active/birth membership arrays AND ordered indices match exactly on BOTH nets. Full reruns use changed centered pair/scalar/birth-beta arithmetic, not a fixed precomputed trajectory. The small difference is amplified f32 carrier roundoff, below the fixed2e-6 bank-fidelity gate. Final normalized feature h max differs4.15444e-5/3.64780e-5; do not claim those features or whole predictions are byte-identical. Time25.313s, peak RSS5,627,199,488bytes. Receipt centered_parent_numerics_v1.json, output SHA256 b77ec029874a66573581ee8118565714f5ea4d06878a1cbd42cdfe269f14ff77.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_pair_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_pair_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_relu_marginals_v1.py](code/astra_phase2_reexamination_v1/centered_relu_marginals_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/check_centered_bank_baseline_v1.py](code/astra_phase2_reexamination_v1/check_centered_bank_baseline_v1.py)
- [code/astra_phase2_reexamination_v1/check_centered_parent_numerics_v1.py](code/astra_phase2_reexamination_v1/check_centered_parent_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)

<details>
<summary>Additional programs and dependencies (10)</summary>

- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [CENTERED_MOMENT_NUMERICS_V1_F64_VERDICT.md#L1](CENTERED_MOMENT_NUMERICS_V1_F64_VERDICT.md#L1)
- [CENTERED_MOMENT_NUMERICS_V1_F64_VERDICT.md#L27](CENTERED_MOMENT_NUMERICS_V1_F64_VERDICT.md#L27)

<details>
<summary>All result files and records (3)</summary>

- [CENTERED_MOMENT_NUMERICS_V1_F64_VERDICT.md](CENTERED_MOMENT_NUMERICS_V1_F64_VERDICT.md) · record
- [centered_parent_numerics_v1.json](centered_parent_numerics_v1.json) · result
- [CENTERED_MOMENT_NUMERICS_DESIGN_V1.md](CENTERED_MOMENT_NUMERICS_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-centered-moment-rewrite-scalar-pair-full-depth-float64-fidelity) · [Copy provenance](../../../PROVENANCE.json)

