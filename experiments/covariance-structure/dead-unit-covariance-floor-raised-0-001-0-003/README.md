# Dead-unit covariance floor raised 0.001 to 0.003 at all layers: 24-seed net -0.037%, aggregate +0.048%, not stacked

2026-10-16T14:18:25Z · experiment · rejected · measured

The dead-unit covariance roster floor (_DCV_FLOOR, the threshold below which dead-unit rows are cut from the covariance sandwich) was raised from 0.001 to 0.003 (DCV3) on the estimator submitted as #336412 (S2tr) candidate. On the 8 screening seeds it passed the pre-registered bar, but the pre-registered 16-seed confirm on the A80 seeds failed: the deeper cut costs about as much accuracy as it saves in bill. The queued 0.004 and 0.005 probes were cancelled and the build pg2dcv3 was left unused.

Recorded measurement:

    8 seeds 7103248-55 vs ctl_s2tr: xMSE +0.023% (SE .344, 5/8), bill -0.270% (8/8, SE .015), xnet -0.248%, aggregate -0.235%. A80 16 seeds 7103080-095: xMSE +0.326% (SE .235, 4/16 lower), bill -0.258% (16/16), xnet +0.068%, aggregate +0.188%; all 24: xMSE +0.225% (SE .192), bill -0.262%, xnet -0.037% (SE .196), aggregate +0.048% (bar xnet <= -0.15%, aggregate < 0).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_passrel_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_pg2dcv3_release.py](code/cand_pg2dcv3_release.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4202](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4202)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4204](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4204)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4211](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4211)

<details>
<summary>All result files and records (3)</summary>

- [fragment_dcv3_v1.py.txt](fragment_dcv3_v1.py.txt) · code
- [BUILD_pg2dcv3.json](BUILD_pg2dcv3.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- rejects: [Dead-unit covariance floor 0.002 at all layers: 24 seeds net -0.082%, bill -0.148%, stacked as a marginal keeper](../dead-unit-covariance-floor-0-002-all-layers-24/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-dead-unit-covariance-floor-raised-0-001-0-003) · [Copy provenance](../../../PROVENANCE.json)

