# Fixed Gaussian/transport prefix sum: gate failed

2026-09-06T02:14Z · experiment · rejected · measured

Phase2, September5,2026. Small four-step dense identity below1.4e-17; rank35 source0..12, same frozen target cell, current host byte-identical. This sum deliberately omitted the non-Gaussian heavy two-edge source and is not a verdict on the complete augmented expansion.

Recorded measurement:

    Phase2, September5,2026. Small four-step dense identity below1.4e-17; rank35 source0..12, same frozen target cell, current host byte-identical. No new MC, state writes, fitting, source/age selection, or score. Fixed natural sum gives weighted signal error ratio .926152227, cosine .279422474, correction/error norm .214390592. Independent ratios .932838256/.923361476. Unweighted ratio .820999305. The .65/.65/.70 cost-motivated gates FAIL. Source slice norm .5058224.8.47s, peak4.776GB. Do not build a sidecar, tune coefficients/ages, or fit this specified sum. Receipt: cumulative_backward_source_r35_l12_v1.json/.npz.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/complete_heavy_path_birth_v1.py](code/astra_phase2_reexamination_v1/complete_heavy_path_birth_v1.py)
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

- [CUMULATIVE_BACKWARD_SOURCE_V1_CLOSED.md#L1](CUMULATIVE_BACKWARD_SOURCE_V1_CLOSED.md#L1)
- [CUMULATIVE_BACKWARD_SOURCE_V1_CLOSED.md#L10](CUMULATIVE_BACKWARD_SOURCE_V1_CLOSED.md#L10)

<details>
<summary>All result files and records (2)</summary>

- [CUMULATIVE_BACKWARD_SOURCE_V1_CLOSED.md](CUMULATIVE_BACKWARD_SOURCE_V1_CLOSED.md) · record
- [cumulative_backward_source_r35_l12_v1.json](cumulative_backward_source_r35_l12_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-fixed-gaussian-transport-prefix-sum-gate-failed) · [Copy provenance](../../../PROVENANCE.json)

