# Layer-15 row cut z lowered 3.5 to 3.0: 24 seeds net -0.315% at bill -0.360%; 2.75 and 2.5 are a cliff

2026-10-16T15:30:55Z · experiment · adopted · measured

The layer-15 row cut (_L15_ROWCUT_Z, the threshold beyond which rows of the final exact layer are dropped) was lowered from 3.5 to 3.0 (L15RC30) on the estimator submitted as #336412 (S2tr): a real bill cut with no measurable accuracy cost on 24 seeds, stacked value-only and shipped in the V-A-noB file (#336570). Going to 2.75 or 2.5 is a cliff. On the independent store networks the accuracy neutrality did not hold (+0.397% on 10 ranks; pooled about -0.18% net).

Recorded measurement:

    L15RC30 (47a24487) 8 seeds: xMSE +0.236% (SE .21, 3/8), bill -0.310%, xnet -0.074%; 24 seeds: xnet -0.315% (SE .121), aggregate -0.376%, 17/24 lower; bill -0.360% (S8 -0.31, A80 -0.385), xMSE +0.045% (SE .117). L15RC275 (10078c04): xMSE +0.958% (SE .58, 4/8), bill -0.508%, xnet +0.450%. L15RC25 (58801757): xMSE +4.682% (SE .53, 0/8), bill -0.674%, xnet +4.007%. Store-32: n10 xMSE +0.397% (SE .16, 1/10), xnet +0.06; n11 xnet +0.003 -> pooled with seeds -0.179% (SE .09).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_ladder2h_v1/cand_s2tr_l2h_l15rc30snap_v1.py](code/opus_ladder2h_v1/cand_s2tr_l2h_l15rc30snap_v1.py)
- [code/opus_ladder2h_v1/cand_s2tr_l2h_l15rc275_v1.py](code/opus_ladder2h_v1/cand_s2tr_l2h_l15rc275_v1.py)
- [code/opus_kinkoracle_v1/cand_decomp_l15rc30_v1.py](code/opus_kinkoracle_v1/cand_decomp_l15rc30_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4226](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4226)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4232](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4232)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4273](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4273)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4275](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4275)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4294](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4294)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4452](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4452)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4453](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4453)

<details>
<summary>All result files and records (4)</summary>

- [CONF24_l15rc30snap_v1.json](CONF24_l15rc30snap_v1.json) · result
- [L2H_l15rc275_v1.json](L2H_l15rc275_v1.json) · result
- [frag_l15rc30_v1.py.txt](frag_l15rc30_v1.py.txt) · code
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Dead-unit covariance floor 0.002 at all layers: 24 seeds net -0.082%, bill -0.148%, stacked as a marginal keeper](../../covariance-structure/dead-unit-covariance-floor-0-002-all-layers-24/README.md)
- shipped in: [Submission #336570: join ordering and three cutoff changes; a public network failed at .430 s residual](../../../submissions/336570/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-layer-15-row-cut-z-lowered-3-5-3) · [Copy provenance](../../../PROVENANCE.json)

