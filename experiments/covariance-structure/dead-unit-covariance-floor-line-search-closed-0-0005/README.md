# Dead-unit covariance floor line search closed: 0.0005 costs +0.250% net, layers 1-9 only cannot buy over 0.06% bill

2026-10-16T16:15:04Z · experiment · rejected · measured

Two further moves of the dead-unit covariance (sandwich) floor were killed: an early-only deeper floor (0.003 or 0.004 at layers 1-9, 0.002 kept at layers 10-15) was killed at its pre-registered seed-1 gate because the whole incremental bill a deeper floor can buy over 0.002 is about 0.12% on the 8 seeds (an early-only cut is at most about 0.06%), below the -0.15% bar; and restoring sandwich rows with a 0.0005 floor hurt accuracy. The sandwich dead floor sits at its accuracy optimum, unlike the push dead cut where restoring rows helps. The only threshold restore in the estimator submitted as #336412 (S2tr) is the dead-cut z 3.5 to 3.0902.

Recorded measurement:

    DCVE3/DCVE4 bill vs DCV2 on seed 7103248: +0.000% (identical flops, 41,062 ops; seed 1 degenerate because the roster rounds to 64). S8 DCV3-vs-DCV2 bills: 0.000 / -0.047 / -0.110 / -0.151 / -0.227 / -0.140 / -0.110 / -0.173% (mean -0.12%). DCV05 (_DCV_FLOOR 0.0005): xMSE +0.180% (SE .08, 0/8 lower), bill +0.069%, xnet +0.250%. Floor ladder: DCVX +0.30, DCV05 +0.18, DCV2 +0.07 xMSE with -0.15% bill.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_dcve_v1/build_dcve_v1.py](code/opus_dcve_v1/build_dcve_v1.py)
- [code/opus_dcve_v1/cand_s2tr_dcve3_v1.py](code/opus_dcve_v1/cand_s2tr_dcve3_v1.py)
- [code/opus_ladder2h_v1/cand_s2tr_l2h_dcv05_v1.py](code/opus_ladder2h_v1/cand_s2tr_l2h_dcv05_v1.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4318](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4318)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4321](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4321)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4325](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4325)

<details>
<summary>All result files and records (3)</summary>

- [PREREG_dcve_v1.md](PREREG_dcve_v1.md) · record
- [L2H_dcv05_v1.json](L2H_dcv05_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- measured by: [Dead-unit covariance floor 0.002 at all layers: 24 seeds net -0.082%, bill -0.148%, stacked as a marginal keeper](../dead-unit-covariance-floor-0-002-all-layers-24/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-dead-unit-covariance-floor-line-search-closed-0-0005) · [Copy provenance](../../../PROVENANCE.json)

