# Learned covariance edge off at layers 8, 9 or 14: +0.085% to +0.133% adjusted on 25-28 ranks; it stays on at every layer

2026-10-19T16:35:03Z · experiment · rejected · measured

The 10-16 per-layer secondaries of the edge-off experiment (8 seeds) had suggested removing the learned covariance edge module at layers 8, 9 and 14 was neutral or helpful at about 0.08% bill per layer; on store ranks switching it off at 8, 9 and 14, at 8 and 9, or at 14 alone is worse every time, so the edge stays on at every layer.

Recorded measurement:

    edge_l1013 (off 8, 9, 14; 28 ranks vs stk1): raw +0.257% (se .21, 10/28 lower), bill -0.168% (adjusted +0.085%); edge_l1014 (off 8-9, 25 ranks): raw +0.219%, bill -0.101% (+0.115%); edge_l813 (off 14, 25 ranks): raw +0.118%, bill +0.016% (+0.133%); seed secondaries L8 0.00 / L9 -0.06 / L13 +0.30 / L14 -0.12; edgel1012 (off 8, 9, 13, 14) +0.62% on store earlier

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_edge_l1013_sci_v1.py](code/candidate_edge_l1013_sci_v1.py)
- [code/candidate_edge_l1014_sci_v1.py](code/candidate_edge_l1014_sci_v1.py)
- [code/candidate_edge_l813_sci_v1.py](code/candidate_edge_l813_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5974](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5974)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5979](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5979)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5984](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5984)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-learned-covariance-edge-off-layers-8-9-or-14) · [Copy provenance](../../../PROVENANCE.json)

