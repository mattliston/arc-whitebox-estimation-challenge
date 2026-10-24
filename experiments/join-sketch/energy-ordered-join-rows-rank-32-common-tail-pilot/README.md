# Energy-ordered join rows with rank-32 common tail: pilot-8 incremental adjusted +0.11%

2026-10-15T02:57:28Z · experiment · rejected · measured

The exact stack of the energy-ordered join rows and the rank-32 compact common-u looked additive on the two fixed networks but did not beat the compact common-u alone on the 8-network pilot, so it was closed and the compact common-u advanced on its own.

Recorded measurement:

    fixed 2: adjusted -1.366528% vs d401, incremental -0.532506% vs UC32, both seeds lower; pilot 8: -0.088805% vs d401, +0.107858% vs UC32

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_jeo_uc32_stack_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_jeo_uc32_HELD_v1.py](code/candidate_jeo_uc32_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3284](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3284)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3287](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3287)

<details>
<summary>All result files and records (3)</summary>

- [DUAL_DIAGNOSTIC_DECISION_v1.json](DUAL_DIAGNOSTIC_DECISION_v1.json) · result
- [CURRENT_PARENT_DIAGNOSTIC_DECISION_v1.json](CURRENT_PARENT_DIAGNOSTIC_DECISION_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Common-coordinate tail rank 32: two-network -0.84%, pilot-8 -0.20%, all-24 adjusted +0.17%](../../third-order-planes/common-coordinate-tail-rank-32-two-network-0-84/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-energy-ordered-join-rows-rank-32-common-tail-pilot) · [Copy provenance](../../../PROVENANCE.json)

