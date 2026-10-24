# Read-column family closed: rank-2 tail off +2.2% bill; tail over all columns -0.33% on 8 seeds, +0.58% on 24

2026-10-16T14:26:11Z · experiment · rejected · measured

Two partial versions of the exact read of the plane bank were tested on the estimator submitted as #336412 (S2tr): READR (the rank-2 read tail RTT2 off, the read-column prefix RC z2 kept) is lossless in accuracy but costs +2.2% bill; READC (the RC z2 prefix off, the rank-2 tail extended over all live columns) passed the 8-seed bar with a rounding-draw-like scatter and then failed its 16-seed A80 confirm. With READX from the morning, the whole read-columns family is closed for accuracy.

Recorded measurement:

    READR (64710e04): xMSE -0.166% (SE .54, 1/8 lower), bill +2.201%, xnet +2.035%. READC (5c8de363): 8 seeds xMSE -0.395% (SE .675, 5/8 lower), bill +0.065%, xnet -0.329%, aggregate -0.189%, per seed -1.53 -1.84 +1.03 -1.07 -2.55 -1.15 +0.70 +3.25; A80 16 seeds xMSE +0.974% (SE .51, 6/16 lower); 24 seeds xMSE +0.517% (SE .42), xnet +0.583%, aggregate +0.738%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_ladder2h_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_s2tr_l2h_readc_v1.py](code/cand_s2tr_l2h_readc_v1.py)
- [code/cand_s2tr_l2h_readr_v1.py](code/cand_s2tr_l2h_readr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4207](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4207)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4227](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4227)

<details>
<summary>All result files and records (4)</summary>

- [L2H_readr_v1.json](L2H_readr_v1.json) · result
- [L2H_readc_v1.json](L2H_readc_v1.json) · result
- [L2H_readc24_v1.json](L2H_readc24_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-read-column-family-closed-rank-2-tail-off-2) · [Copy provenance](../../../PROVENANCE.json)

