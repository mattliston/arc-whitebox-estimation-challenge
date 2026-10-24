# Late missing third sector is carried-dominated at the fixed cell

2026-09-06T02:06Z · experiment · informational · measured

Phase2, September5,2026. Fixed rank35 pre/post12 -> pre13, q4 and32 rows. This does not reverse any single-cell source failure or prove that a given omitted source is useful cumulatively.

Recorded measurement:

    Phase2, September5,2026. Fixed rank35 pre/post12 -> pre13, q4 and32 rows. Two independent8M streams; no fit, new seed/count/layer or source re-score. Both new pre-slice runs reproduce old post means BYTE EXACTLY. Small central-third certificate1.12e-13, diagonal arithmetic consistency4.17e-7. The consumer decomposition is linear to2.35e-17, and reproduces the original total oracle correction BYTE EXACTLY. Two-stream consumer-weighted energy point estimates: total9.79866652e-10, carried7.33191369e-10 (74.83% of total), fresh1.08379265e-10 (11.06%). The two components are correlated; their cross contribution supplies the remaining14.11%. These fractions are NOT additive attribution shares and are not confidence bounds. Carried/total cosine .946598794. The predeclared carried-dominance gate (fresh<=25%, cosine>=.8) PASSES. This does not reverse any single-cell source failure or prove that a given omitted source is useful cumulatively. It justifies a separately designed fixed prefix-propagation test, not more last-layer source tuning. No raw/adjusted score, candidate, panel or submission. All jobs complete. Receipt: third_error_innovation_r35_l12_v1.json/.npz and both pre-slice stream receipts. The official bank remains329926 at4.98223146e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/analyze_third_error_innovation_v1.py](code/astra_phase2_reexamination_v1/analyze_third_error_innovation_v1.py)
- [code/astra_phase2_reexamination_v1/check_cumulative_backward_source_v1.py](code/astra_phase2_reexamination_v1/check_cumulative_backward_source_v1.py)
- [code/astra_phase2_reexamination_v1/cumulative_backward_source_v1.py](code/astra_phase2_reexamination_v1/cumulative_backward_source_v1.py)
- [code/astra_phase2_reexamination_v1/full_edge_gaussian_bond3_v1.py](code/astra_phase2_reexamination_v1/full_edge_gaussian_bond3_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/linear_transport_residual_source_v1.py](code/astra_phase2_reexamination_v1/linear_transport_residual_source_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py](code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [THIRD_ERROR_INNOVATION_V1_VERDICT.md#L1](THIRD_ERROR_INNOVATION_V1_VERDICT.md#L1)
- [THIRD_ERROR_INNOVATION_V1_VERDICT.md#L12](THIRD_ERROR_INNOVATION_V1_VERDICT.md#L12)

<details>
<summary>All result files and records (3)</summary>

- [THIRD_ERROR_INNOVATION_V1_VERDICT.md](THIRD_ERROR_INNOVATION_V1_VERDICT.md) · record
- [third_error_innovation_r35_l12_v1.json](third_error_innovation_r35_l12_v1.json) · result
- [THIRD_ERROR_INNOVATION_ATTRIBUTION_DESIGN_V1.md](THIRD_ERROR_INNOVATION_ATTRIBUTION_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-late-missing-third-sector-carried-dominated-fixed-cell) · [Copy provenance](../../../PROVENANCE.json)

