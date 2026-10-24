# Learned edge restricted to layers 10-12: -0.23% net on 8 seeds, +0.62% MSE on 21 store networks, killed

2026-10-16T18:59:55Z · experiment · rejected · measured

Because the per-layer secondaries of the edge-off arm suggested layers 8-9 and 13-14 carry nothing, the learned covariance edge was restricted to layers 10-12 (a text edit of the three layer gates). The 8-seed read was a pass (bill -0.313%), the store networks reversed it.

Recorded measurement:

    edgel1012 (7f5224fd) S8 vs ctl_all24: aggregate xMSE +0.082%, bill -0.313%, score -0.231% (n 8); store n21 xMSE +0.62% -> FAIL.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_solo_v1/cand_edgel1012.py](code/opus_solo_v1/cand_edgel1012.py)
- [code/opus_kinkoracle_v1/cand_decomp_edgel1012_v1.py](code/opus_kinkoracle_v1/cand_decomp_edgel1012_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4450](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4450)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4462](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4462)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4467](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4467)

<details>
<summary>All result files and records (2)</summary>

- [GRADE_edgel1012.txt](GRADE_edgel1012.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-learned-edge-restricted-layers-10-12-0-23-net) · [Copy provenance](../../../PROVENANCE.json)

