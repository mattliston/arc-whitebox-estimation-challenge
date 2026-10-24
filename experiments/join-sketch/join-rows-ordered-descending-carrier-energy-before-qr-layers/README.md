# Join rows ordered by descending carrier energy before the QR at layers 10/13 (zero bill): fixed-2 -0.26%, pilot-8 -0.12%

2026-10-15T02:37:52Z · experiment · inconclusive · measured

The record found that the Nystrom-selected join row ids were index-sorted before the QR despite a comment promising energy order. Sorting the same selected set by descending carrier energy at layers 10 and 13 (JEO, layer 7 unchanged) costs nothing and gained on the two fixed networks and on the 8-network pilot, so it was stacked with the compact common-u arm rather than confirmed alone.

Recorded measurement:

    fixed 2: adjusted -0.259075259%, MSE -0.259055806%, bill -0.000019504% (7100136 -0.053617234%, 7100137 -0.464110929%); pilot 8: adjusted -0.122189% (4/8 lower)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_join_energy_order_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_energy_order_HELD_v1.py](code/candidate_energy_order_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3275](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3275)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3276](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3276)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3279](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3279)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3281](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3281)

<details>
<summary>All result files and records (3)</summary>

- [JEO_PILOT8_DECISION_v1.json](JEO_PILOT8_DECISION_v1.json) · result
- [OWNER_RETIREMENT_v1.json](OWNER_RETIREMENT_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Common-coordinate tail rank 32: two-network -0.84%, pilot-8 -0.20%, all-24 adjusted +0.17%](../../third-order-planes/common-coordinate-tail-rank-32-two-network-0-84/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-join-rows-ordered-descending-carrier-energy-before-qr-layers) · [Copy provenance](../../../PROVENANCE.json)

