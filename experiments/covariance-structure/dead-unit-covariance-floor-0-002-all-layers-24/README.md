# Dead-unit covariance floor 0.002 at all layers: 24 seeds net -0.082%, bill -0.148%, stacked as a marginal keeper

2026-10-16T15:10:09Z · experiment · adopted · measured

The dead-unit covariance roster floor (_DCV_FLOOR) raised from 0.001 to 0.002 (DCV2) on the estimator submitted as #336412 (S2tr). The bill cut is consistent on all 24 seeds but about half of it is eaten by an accuracy cost inside the noise. It was stacked with the layer-15 row cut (exactly additive on 24 seeds) and entered the clean stacks and the fired V-A-noB file (#336570).

Recorded measurement:

    8 seeds: xMSE +0.018% (SE .169, 4/8), bill -0.150% (8/8), xnet -0.133% (SE .183). 24 seeds (S8 + A80): xnet -0.082% (SE .127), aggregate -0.048%, 13/24 lower; bill -0.148% (24/24), xMSE +0.067% (SE .122). Stack with L15 row cut 3.0 on 24 seeds: xnet -0.397% (SE .15), aggregate -0.424%, 17/24 lower; bill -0.509%, xMSE +0.112% (SE .14); singles -0.082 + -0.315. Boundary requests of the stack vs S2tr: warm 45,514 vs 45,646 (-132), cold first 47,372 vs 47,336 (+36); warm flops 243.27e9 vs 245.03e9 (-0.72%).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_ladder2h_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_s2tr_l2h_dcv2snap_v1.py](code/cand_s2tr_l2h_dcv2snap_v1.py)
- [code/cand_s2tr_l2h_stkdcv2rc30_v1.py](code/cand_s2tr_l2h_stkdcv2rc30_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4202](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4202)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4273](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4273)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4289](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4289)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4295](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4295)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4297](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4297)

<details>
<summary>All result files and records (4)</summary>

- [CONF24_dcv2snap_v1.json](CONF24_dcv2snap_v1.json) · result
- [CONF24_stkdcv2rc30_v1.json](CONF24_stkdcv2rc30_v1.json) · result
- [frag_dcv2_v1.py.txt](frag_dcv2_v1.py.txt) · code
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Layer-15 row cut z lowered 3.5 to 3.0: 24 seeds net -0.315% at bill -0.360%; 2.75 and 2.5 are a cliff](../../late-layer-exactness/layer-15-row-cut-z-lowered-3-5-3/README.md)
- shipped in: [Submission #336570: join ordering and three cutoff changes; a public network failed at .430 s residual](../../../submissions/336570/README.md)
- ← measured by: [Dead-unit covariance floor line search closed: 0.0005 costs +0.250% net, layers 1-9 only cannot buy over 0.06% bill](../dead-unit-covariance-floor-line-search-closed-0-0005/README.md)
- ← rejects: [Dead-unit covariance floor raised 0.001 to 0.003 at all layers: 24-seed net -0.037%, aggregate +0.048%, not stacked](../dead-unit-covariance-floor-raised-0-001-0-003/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-dead-unit-covariance-floor-0-002-all-layers-24) · [Copy provenance](../../../PROVENANCE.json)

