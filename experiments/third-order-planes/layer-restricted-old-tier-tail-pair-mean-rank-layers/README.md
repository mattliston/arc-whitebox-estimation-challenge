# Layer-restricted old-tier tail and pair-mean rank at layers 12-14: rates stay 0.7 to 1.0, all arms below the bar

2026-10-16T15:24:36Z · experiment · rejected · measured

Restricting the old-tier tail growth (z 3.0) or the pair-mean rank growth (LCR 384) to the last layers before the readout did not change their rate: OTT30 at layers 13-14 kept the same about 0.7 accuracy-per-bill rate as at all layers and failed the bar; OTT30 at 12-14, OTTP30 at 13-14 and LCR384 at 13-14 were killed on their seed-1 price because the family rate cannot reach the bar. The LCR family as a whole cannot pass because the full effect of removing the pair-mean truncation at layers 10-14 is only -0.36% at the final layer.

Recorded measurement:

    OTT30 at L13-14 (12817bbc): xMSE -0.272% (SE .24, 5/8 lower), bill +0.373%, xnet +0.101%, aggregate +0.085%; per layer L14 -0.49 (.24), L15 -0.27. OTT30 at L12-14 priced +0.615% on seed 1 (needs <= -0.77%); LCR384 at L13-14 priced +1.21%; OTTP30 at L13-14 priced +0.278% (needs rate >= 1.55; OTT rates measured 0.7-1.0).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_ladder2h_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_s2tr_l2h_ott30_1314_v1.py](code/cand_s2tr_l2h_ott30_1314_v1.py)
- [code/cand_s2tr_l2h_lcr1314x_v1.py](code/cand_s2tr_l2h_lcr1314x_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4283](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4283)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4289](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4289)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4312](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4312)

<details>
<summary>All result files and records (2)</summary>

- [L2H_ott30_1314_v1.json](L2H_ott30_1314_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-layer-restricted-old-tier-tail-pair-mean-rank-layers) · [Copy provenance](../../../PROVENANCE.json)

