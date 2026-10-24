# Common-coordinate tail rank 32: two-network -0.84%, pilot-8 -0.20%, all-24 adjusted +0.17%

2026-10-15T03:14:28Z · experiment · rejected · measured

The rank-32 compact common-u (224 kept coordinates plus a rank-32 joint tail in the 256 budget) was the strongest two-network and pilot survivor of the night, but the full 24-network science failed on out-of-sample networks. Adding the omitted common coordinates 320 to 384 at layer 13 into the same budget (full-tail 32) was worse than UC32 itself.

Recorded measurement:

    UC32 fixed 2: adjusted -0.838487%, MSE -1.040508%, bill +0.204145%; pilot 8: -0.196451% (6/8 lower; MSE -0.410679%, bill +0.215112%); full 24: fresh 16 +0.078549%, OOS 8 +0.360605%, all 24 +0.172480% adjusted (MSE -0.041681%, bill +0.214250%); full-tail 32 fixed 2 vs UC32 +0.106585405% (MSE +0.045617256%, bill +0.060940349%), vs d401 -0.732795403%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
- [code/astra_common_u_compact32_v1/check_and_price_uc32_v2.py](code/astra_common_u_compact32_v1/check_and_price_uc32_v2.py)
- [code/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py](code/astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py)
- [code/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3279](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3279)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3281](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3281)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3282](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3282)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3287](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3287)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3291](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3291)

<details>
<summary>All result files and records (3)</summary>

- [PARALLEL_DIAGNOSTIC_DECISION_v1.json](PARALLEL_DIAGNOSTIC_DECISION_v1.json) · result
- [FULLTAIL32_TARIFF_v1.json](FULLTAIL32_TARIFF_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Energy-ordered join rows with rank-32 common tail: pilot-8 incremental adjusted +0.11%](../../join-sketch/energy-ordered-join-rows-rank-32-common-tail-pilot/README.md)
- enables: [Squared right-coefficient weighting of rank-32 common tail: pilot-8 -0.21%, all-24 adjusted +0.14%](../squared-right-coefficient-weighting-rank-32-common-tail-pilot/README.md)
- ← enables: [Join rows ordered by descending carrier energy before the QR at layers 10/13 (zero bill): fixed-2 -0.26%, pilot-8 -0.12%](../../join-sketch/join-rows-ordered-descending-carrier-energy-before-qr-layers/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-common-coordinate-tail-rank-32-two-network-0-84) · [Copy provenance](../../../PROVENANCE.json)

