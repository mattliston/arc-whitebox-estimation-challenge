# Encoder-weighted output projection and triangle terms: adjusted +0.04%/+0.18%

2026-10-14T23:37:07Z · experiment · rejected · measured

Two more two-network smokes closed: weighting the output-projection PCA by the encoder column norms (GOPW) changed nothing, and the triangle arm (three active-core hooks with the exact repeated-index subtraction p^3 - 3pq + 2r in float64, one-hop diagonal) was worse on both networks.

Recorded measurement:

    GOPW: mean log net +0.0004184080338714121, adjusted +0.04184955787222134%; triangle: adjusted +0.19667399501875288% (7100136) and +0.1562006700796923% (7100137), mean +0.17643528854305213%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_encoder_weighted_gop2_v1/fragment_encoder_weighted_gop2_HELD_v1.py](code/astra_encoder_weighted_gop2_v1/fragment_encoder_weighted_gop2_HELD_v1.py)
- [code/astra_triangle_cycle_v1/fragment_triangle_HELD_v1.py](code/astra_triangle_cycle_v1/fragment_triangle_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3182](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3182)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3183](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3183)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3196](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3196)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3200](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3200)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3204](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3204)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3205](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3205)

<details>
<summary>All result files and records (3)</summary>

- [ROOT_DIAGNOSTIC_CLOSURE_v1.json](ROOT_DIAGNOSTIC_CLOSURE_v1.json) · result
- [ROOT_DIAGNOSTIC_RETIREMENT_v1.json](ROOT_DIAGNOSTIC_RETIREMENT_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-encoder-weighted-output-projection-triangle-terms-adjusted-0-04) · [Copy provenance](../../../PROVENANCE.json)

