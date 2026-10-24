# Pair-mean u-truncation removed at layers 10-14 (rank 256 to 384): -0.36% MSE for +2.5% bill; ordering recovers most

2026-10-16T14:43:38Z · experiment · rejected · measured

LCRX removed the pair-mean u-truncation at layers 10-14 (_LCR_RU 256 to 384, the u-leg of the u/v pair-mean split carried on the leading 256 frame coordinates). It bought -0.36% final MSE for +2.5% bill, with the cost concentrated at layers 13-14. Because the energy-ordered Nystrom rows (JORD, zero bill) already recover about -0.24% of it on 24 seeds, a better ordering has at most about 0.1% left, and the u/v ordering line was closed. The shallower cut UVP192 (r_u 256 to 192 at layer 15) also missed the bar.

Recorded measurement:

    LCRX (97f6b555): xMSE -0.359% (SE .42, 5/8 lower), bill +2.502%, xnet +2.144%; per layer L11 -0.59 (.24), L12 -0.67, L13 -0.81 (.29), L14 -0.79 (.37), L15 -0.36 (.42). UVP192 (0b5ba92c): xMSE +0.193% (SE .09, 3/8), bill -0.074%, xnet +0.119%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_ladder2h_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_s2tr_l2h_lcrx_v1.py](code/cand_s2tr_l2h_lcrx_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4226](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4226)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4247](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4247)

<details>
<summary>All result files and records (2)</summary>

- [L2H_lcrx_v1.json](L2H_lcrx_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-pair-mean-u-truncation-removed-layers-10-14-rank) · [Copy provenance](../../../PROVENANCE.json)

