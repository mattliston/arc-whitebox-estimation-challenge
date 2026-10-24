# More depth-seven kernel routes add +0.028% FLOPs, 325 requests per network and +30% residual time

2026-10-19T17:24:29Z · experiment · rejected · measured

Setting the depth-7 route gain to 0 (take every depth-7 kernel route the fitted bill model favors) buys no flops and costs requests and residual time: the marginal depth-7 routes are dead and the kernel depth table stays as the engineers left it.

Recorded measurement:

    hz7_gain0 (30 ranks vs stk1): bill +0.028% (~-0.05% vs the #337126 physics), raw -0.03% (byte-level), requests +325 median (+1,090 max), residual +0.082 s median (+30%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_hz7_gain0_sci_v1.py](code/candidate_hz7_gain0_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5993](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5993)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-more-depth-seven-kernel-routes-add-0-028-flops) · [Copy provenance](../../../PROVENANCE.json)

