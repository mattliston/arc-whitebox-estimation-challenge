# Full-depth transport shadow rejected; localized suffix is a separate gate

2026-09-05T17:35Z · experiment · informational · measured

Full finite-difference half-step disagreement is 7.43% / 6.15% at layer 4, 2.09% / 2.99% at L9, and .21% / .40% at L13. Shadow adjoint relative errors are 0.0056% / 0.311%. The failure is localized in elapsed propagation depth.

Recorded measurement:

    | Perturbed layer | Rank 35 | Rank 100 | | --- | --- | --- | | 4 | 0.335184 | 0.293119 | | 9 | 0.081706 | 0.078488 | | 13 | 0.009691 | 0.010204 | Full finite-difference half-step disagreement is 7.43% / 6.15% at L4, 2.09% / 2.99% at L9, and .21% / .40% at L13. Shadow adjoint relative errors are 0.0056% / 0.311%. No targets or fits were used. Receipt: current_transport_shadow_fidelity_v3.json. Full-depth fidelity fails the predeclared gate, and no full-depth fit will be launched on this shadow. The failure is localized in elapsed propagation depth. A distinct follow-up may freeze the entire prefix through layer 8 and train only layers 9..15. Before any such fit, test one fixed simultaneous random correction across layers 9..14 on both ranks, epsilon 2e-4 and half epsilon. Require <=10% response error and <=5% full/shadow half-step discrepancy on each rank. This checks the suffix as a coupled object, not just isolated endpoints. It does not relax the failed full-depth gate or establish an accuracy gain.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/check_current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/check_current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/check_current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v2.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v2.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/generate_current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/generate_current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/generate_current_transport_shadow_v2.py](code/astra_phase2_reexamination_v1/generate_current_transport_shadow_v2.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)

<details>
<summary>Additional programs and dependencies (4)</summary>

- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [CURRENT_TRANSPORT_SHADOW_V3_VERDICT.md#L1](CURRENT_TRANSPORT_SHADOW_V3_VERDICT.md#L1)
- [CURRENT_TRANSPORT_SHADOW_V3_VERDICT.md#L14](CURRENT_TRANSPORT_SHADOW_V3_VERDICT.md#L14)
- [CURRENT_TRANSPORT_SHADOW_V3_VERDICT.md#L18](CURRENT_TRANSPORT_SHADOW_V3_VERDICT.md#L18)

<details>
<summary>All result files and records (4)</summary>

- [CURRENT_TRANSPORT_SHADOW_V3_VERDICT.md](CURRENT_TRANSPORT_SHADOW_V3_VERDICT.md) · record
- [current_transport_shadow_fidelity_v3.json](current_transport_shadow_fidelity_v3.json) · result
- [CURRENT_TRANSPORT_SHADOW_V1.md](CURRENT_TRANSPORT_SHADOW_V1.md) · record
- [CURRENT_TRANSPORT_SHADOW_V2_REPAIR.md](CURRENT_TRANSPORT_SHADOW_V2_REPAIR.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-full-depth-transport-shadow-rejected-localized-suffix-separate-gate) · [Copy provenance](../../../PROVENANCE.json)

