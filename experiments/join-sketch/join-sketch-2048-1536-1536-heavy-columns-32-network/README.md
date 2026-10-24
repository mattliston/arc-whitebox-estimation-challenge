# Join sketch on 2048/1536/1536 heavy columns: 32-network adjusted and bill both -1.653%

2026-10-18T01:57:17Z · experiment · adopted · measured

The heavy-column Nyström sketch was reduced to 2048, 1536 and 1536 columns at layers 7, 10 and 13. The full-store comparison preserved raw error while cutting bill and became submission #336854.

Recorded measurement:

    7 ranks: adjusted -1.837% (SE .231), raw -0.153%, bill -1.687%; 32 ranks: adjusted -1.653% (SE .104), raw +0.000% (17/32 lower), bill -1.653%; median requests 41,917; residual median .2228, p90 .2271, max .2328.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_v53j.py](code/candidate_v53j.py)
- [code/jsk/make_joinsketch_v6.py](code/jsk/make_joinsketch_v6.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5051](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5051)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5063](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5063)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5067](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5067)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [GRADE_STORE_v1.json](GRADE_STORE_v1.json) · result
- [GRADE_STORE_v1-afe2ae26.json](GRADE_STORE_v1-afe2ae26.json) · result

</details>

## Related work

- shipped in: [Submission #336854: heavy-column Nyström join sketch: 100 networks passed at 1.4517480806360376e-09](../../../submissions/336854/README.md)

[Research graph](../../../graph/index.html#node=2026-10-18-join-sketch-2048-1536-1536-heavy-columns-32-network) · [Copy provenance](../../../PROVENANCE.json)

