# Chi-radius trace-preserving fourth shape: algebra qualified, no value

2026-09-06T18:47Z · finding · informational · measured

- fourth-cumulant two-pair plane (K22)/fourth-cumulant triple-single plane (K31)/marginal fourth maximum error1.11e-16. - Trace relative error1.81e-16; angular scalar recovery1.39e-17. - Both plane diagonals agree within5.55e-17. - Radial cancellation and zero-mean/zero-Gaussian-input identities exact. - The new nonlocal K4-to-K3 source identity agrees within1.69e-21. - Runtime0.0182seconds, peak38,125,568bytes; no worker remains live. The plane constructor and fixed trace fit require only O(n^2) work.

Recorded measurement:

    - K22/K31/marginal fourth maximum error1.11e-16. - Trace relative error1.81e-16; angular scalar recovery1.39e-17. - Both plane diagonals agree within5.55e-17. - Radial cancellation and zero-mean/zero-Gaussian-input identities exact. - The new nonlocal K4-to-K3 source identity agrees within1.69e-21. - Runtime0.0182seconds, peak38,125,568bytes; no worker remains live. The plane constructor and fixed trace fit require only O(n^2) work. An80*n^2/layer conservative elementwise estimate gives1.26e9 FLOPs over 15 pair layers, below.3% of a490e9 bank bill; no measured flopscope or runtime result exists. Exact supplied K31 must replace the old radial K31 formula in every pair response that consumes it.

## Programs

- [certify_chi_radius_trace_shape_v1.py](certify_chi_radius_trace_shape_v1.py)
- [../../../experiments/fourth-order-terms/trace-dependent-variancemean-resummation-fixed-formal-consumer-closed/code/astra_phase2_reexamination_v1/check_chi_radius_trace_shape_value_v1.py](../../../experiments/fourth-order-terms/trace-dependent-variancemean-resummation-fixed-formal-consumer-closed/code/astra_phase2_reexamination_v1/check_chi_radius_trace_shape_value_v1.py)
- [../../../experiments/fourth-order-terms/trace-dependent-variancemean-resummation-fixed-formal-consumer-closed/code/astra_phase2_reexamination_v1/chi_radius_trace_shape_v1.py](../../../experiments/fourth-order-terms/trace-dependent-variancemean-resummation-fixed-formal-consumer-closed/code/astra_phase2_reexamination_v1/chi_radius_trace_shape_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py)
- [radial_noncentral.py](radial_noncentral.py)
- [../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

## Evidence

- [CHI_RADIUS_TRACE_SHAPE_ALGEBRA_VERDICT_V1.md#L1](CHI_RADIUS_TRACE_SHAPE_ALGEBRA_VERDICT_V1.md#L1)

<details>
<summary>All result files and records (3)</summary>

- [CHI_RADIUS_TRACE_SHAPE_ALGEBRA_VERDICT_V1.md](CHI_RADIUS_TRACE_SHAPE_ALGEBRA_VERDICT_V1.md) · record
- [CHI_RADIUS_TRACE_SHAPE_DESIGN_V1.md](CHI_RADIUS_TRACE_SHAPE_DESIGN_V1.md) · record
- [chi_radius_trace_shape_certificate_v1.json](chi_radius_trace_shape_certificate_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-chi-radius-trace-preserving-fourth-shape-algebra-qualified-no) · [Copy provenance](../../../PROVENANCE.json)

