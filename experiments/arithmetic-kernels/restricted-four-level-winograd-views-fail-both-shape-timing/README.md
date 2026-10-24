# Restricted four-level Winograd views fail both-shape timing gate despite numerical consistency

2026-09-08 · experiment · rejected · measured

Restricting arena views preserved arithmetic but did not reach the required ten-percent timing improvement.

Recorded measurement:

    VERDICT_2026_09_08_PDT.md:
    # Restricted D4 views: numerical consistency passes, timing gate fails
    1 matmul and the predetermined complete KERNEL bill. All public-operation
    |1024,832,8192|.10950174974277616|.11568529112264514|1.0564697951803907|1.0588303637049927|
    |832,8192,832|.1277810619212687|.12281343713402748|.9611239356399934|.9610505601097808|
    Both shapes fail the required <=.90 on both timing ratios. This is a
    bindings cost469,863,680 /438,129,120F restricted, versus524,226,560 /
    `run()` wall1.9509516665712s excludes interpreter/module imports and
    Official329926 remains4.982231460183304e-9 adjusted; qualified local14

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_d4_restricted_views_v1/run_compare.py](code/current_d4_restricted_views_v1/run_compare.py)
- [code/packed_arena_generic.py](code/packed_arena_generic.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

## Evidence

- [VERDICT_2026_09_08_PDT.md#L3](VERDICT_2026_09_08_PDT.md#L3)
- [VERDICT_2026_09_08_PDT.md#L1](VERDICT_2026_09_08_PDT.md#L1)
- [VERDICT_2026_09_08_PDT.md#L12](VERDICT_2026_09_08_PDT.md#L12)
- [VERDICT_2026_09_08_PDT.md#L20](VERDICT_2026_09_08_PDT.md#L20)
- [VERDICT_2026_09_08_PDT.md#L21](VERDICT_2026_09_08_PDT.md#L21)

<details>
<summary>All result files and records (3)</summary>

- [VERDICT_2026_09_08_PDT.md](VERDICT_2026_09_08_PDT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [result.json](result.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-restricted-four-level-winograd-views-fail-both-shape-timing) · [Copy provenance](../../../PROVENANCE.json)

