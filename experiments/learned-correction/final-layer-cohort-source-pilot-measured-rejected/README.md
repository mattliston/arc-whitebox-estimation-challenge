# Final-layer cohort-source pilot: measured and rejected

2026-09-06T07:04Z · experiment · rejected · measured

September6,2026 (result produced September5,23:54PDT). Exact signed-pair identity certified to4.96e-16 relative; both smoke rows and all24 pilot rows passed complete original-prefix/head byte parity and scalar sums. Fails the .90/.92/3-win gate.

Recorded measurement:

    September6,2026 (result produced September5,23:54PDT). Exact signed-pair identity certified to4.96e-16 relative; both smoke rows and all24 pilot rows passed complete original-prefix/head byte parity and scalar sums. Fixed20train/4held-out,40 epochs, matched158-input Q arms, original1e9 references. NumPy final raw error with32 source channels is1.0073885467 times the bank and1.0036697103 times the zero-feature control,0/4 bank wins. NumPy/MPS head differences <=4.366e-11. See final_cohort_source_probe_verdict_v1.json for every net and payload hash. Fails the .90/.92/3-win gate. No200 fit, age/component subset, new epoch, seed/LR, carrier action, candidate or upload. No new adjusted score.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/certify_final_cohort_sources_v1.py](code/astra_phase2_reexamination_v1/certify_final_cohort_sources_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/final_cohort_source_reference_v1.py](code/astra_phase2_reexamination_v1/final_cohort_source_reference_v1.py)
- [code/astra_phase2_reexamination_v1/final_cohort_sources_v1.py](code/astra_phase2_reexamination_v1/final_cohort_sources_v1.py)
- [code/astra_phase2_reexamination_v1/final_variance_source_reference_v1.py](code/astra_phase2_reexamination_v1/final_variance_source_reference_v1.py)
- [code/astra_phase2_reexamination_v1/final_variance_sources_v1.py](code/astra_phase2_reexamination_v1/final_variance_sources_v1.py)
- [code/astra_phase2_reexamination_v1/generate_final_cohort_source_rows_v1.py](code/astra_phase2_reexamination_v1/generate_final_cohort_source_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_final_variance_source_row_v1.py](code/astra_phase2_reexamination_v1/generate_final_variance_source_row_v1.py)
- [code/astra_phase2_reexamination_v1/generate_final_variance_source_rows_v1.py](code/astra_phase2_reexamination_v1/generate_final_variance_source_rows_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)

<details>
<summary>Additional programs and dependencies (10)</summary>

- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/train_final_cohort_source_probe_v1.py](code/astra_phase2_reexamination_v1/train_final_cohort_source_probe_v1.py)
- [code/astra_phase2_reexamination_v1/train_final_variance_source_probe_v1.py](code/astra_phase2_reexamination_v1/train_final_variance_source_probe_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [FINAL_COHORT_SOURCE_PROBE_V1_CLOSED.md#L1](FINAL_COHORT_SOURCE_PROBE_V1_CLOSED.md#L1)
- [FINAL_COHORT_SOURCE_PROBE_V1_CLOSED.md#L10](FINAL_COHORT_SOURCE_PROBE_V1_CLOSED.md#L10)

<details>
<summary>All result files and records (3)</summary>

- [FINAL_COHORT_SOURCE_PROBE_V1_CLOSED.md](FINAL_COHORT_SOURCE_PROBE_V1_CLOSED.md) · record
- [final_cohort_source_probe_verdict_v1.json](final_cohort_source_probe_verdict_v1.json) · result
- [FINAL_COHORT_SOURCE_PROBE_DESIGN_V1.md](FINAL_COHORT_SOURCE_PROBE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-final-layer-cohort-source-pilot-measured-rejected) · [Copy provenance](../../../PROVENANCE.json)

