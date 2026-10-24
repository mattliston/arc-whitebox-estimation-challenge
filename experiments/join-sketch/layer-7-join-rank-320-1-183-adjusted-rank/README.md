# Layer-7 join rank 320: +1.183% adjusted; rank 352 saves effectively no FLOPs

2026-10-20T14:41:20Z · experiment · rejected · measured

The earlier one-network join-rank gain did not generalize to the full 32-network judge. Rank 320 increased MSE +2.118% for -0.907% FLOPs, and rank 352 was blocked by kernel padding.

Recorded measurement:

    - [fable-lead] 2026-10-20T14:41:20Z LAYER-7 JOIN RANK CUT DEAD on 32 ranks (r7_320 3c412296 vs science32_rel_v4a_m15): adjusted +1.183% (se 0.379, lower 14/32) = raw +2.118% + bill -0.907%; the 10-18 one-net read (adjusted -0.23%) did not generalize (a few ranks explode). r7_352 saves no bill (net 35 util 0.108001 vs 0.108016: kernel padding), r7_288 killed at part 8 (strictly worse). The join-rank axis is closed in both directions at every join. Lanes free at 2026-10-20T14:41:20Z -> in-process evals of the common-mode arm (cmode) and the agent's DMT arm start automatically.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_r7_288_sci_v1.py](code/candidate_r7_288_sci_v1.py)
- [code/candidate_r7_320_sci_v1.py](code/candidate_r7_320_sci_v1.py)
- [code/candidate_r7_352_sci_v1.py](code/candidate_r7_352_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6186](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6186)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-layer-7-join-rank-320-1-183-adjusted-rank) · [Copy provenance](../../../PROVENANCE.json)

