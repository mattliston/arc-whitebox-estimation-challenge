# Shared scalar layer-amplitude calibration: sensitivity gate closed

2026-09-05T19:23Z · experiment · rejected · measured

September5,2026, Phase2. Small action certificate passed: repeated-plane errors below1.4e-15 and distinct-only action errors below1.8e-15. One shared bounded(+-.10) least-squares coefficient vector, using the two existing1e9 development targets, predicts raw ratios .972988/.985448, only2.70%/1.46% improvement.

Recorded measurement:

    September5,2026, Phase2. Small action certificate passed: repeated-plane errors below1.4e-15 and distinct-only action errors below1.8e-15. Both current-bank baseline trajectories were byte-identical. Fixed central differences+-.02 covered14 layer coefficients on ranks35/100, two nice10 workers with3BLAS threads,~189seconds each. No training labels entered the response generation. One shared bounded(+-.10) least-squares coefficient vector, using the two existing1e9 development targets, predicts raw ratios .972988/.985448, only2.70%/1.46% improvement. The optimum is interior (offsets roughly -.048..+.010), not artificially limited by the bound. This fails the predeclared20%-on-each information gate. No nonlinear candidate check, 24-net fit, amplitude/window sweep, panel or upload was run. These are linear-response predictions on development nets, NOT validated gains.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_distinct_birth_amplitude_v1.py](code/astra_phase2_reexamination_v1/check_distinct_birth_amplitude_v1.py)
- [code/astra_phase2_reexamination_v1/distinct_birth_amplitude_v1.py](code/astra_phase2_reexamination_v1/distinct_birth_amplitude_v1.py)
- [code/astra_phase2_reexamination_v1/fit_distinct_birth_sensitivity_v1.py](code/astra_phase2_reexamination_v1/fit_distinct_birth_sensitivity_v1.py)
- [code/astra_phase2_reexamination_v1/generate_distinct_birth_sensitivity_v1.py](code/astra_phase2_reexamination_v1/generate_distinct_birth_sensitivity_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [DISTINCT_BIRTH_AMPLITUDE_V1_CLOSED.md#L1](DISTINCT_BIRTH_AMPLITUDE_V1_CLOSED.md#L1)

<details>
<summary>All result files and records (4)</summary>

- [DISTINCT_BIRTH_AMPLITUDE_V1_CLOSED.md](DISTINCT_BIRTH_AMPLITUDE_V1_CLOSED.md) · record
- [distinct_birth_amplitude_certificate_v1.json](distinct_birth_amplitude_certificate_v1.json) · result
- [distinct_birth_sensitivity_verdict_v1.json](distinct_birth_sensitivity_verdict_v1.json) · result
- [DISTINCT_BIRTH_AMPLITUDE_DESIGN_V1.md](DISTINCT_BIRTH_AMPLITUDE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-shared-scalar-layer-amplitude-calibration-sensitivity-gate-closed) · [Copy provenance](../../../PROVENANCE.json)

