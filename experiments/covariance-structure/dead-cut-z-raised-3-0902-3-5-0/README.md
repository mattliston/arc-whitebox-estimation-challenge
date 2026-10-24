# Dead-cut z raised 3.0902 to 3.5: -0.234% MSE for +0.065% bill on 24 seeds (rate 3.6); 4.0, 4.5 and off only add bill

2026-10-16T16:19:32Z · experiment · adopted · measured

The push dead-cut threshold (_DEADCUT_Z) was restored from 3.0902 to 3.5 (DC35) on the estimator submitted as #336412 (S2tr), the only threshold restore in the file. It was the one knob of the day trading better than 1:1 and entered the stacks and the fired V-A-noB (#336570). Mapping the global curve showed accuracy saturating by 3.5-4.0 while bill keeps rising, and a late extension (4.0 at layers 12-14) adds nothing over 3.5. On the independent store networks DC35 read about zero (n6 +0.03%).

Recorded measurement:

    DC35 (1b4c5545) 24 seeds: xMSE -0.234% (SE .13, 15/24 lower), bill +0.065%, xnet -0.169% (SE .13); pooled aggregate -0.133% (S8 -0.052, A80 -0.172). Curve on S8 (xMSE / bill / xnet): 3.5 ~-0.12 / +0.065 / -0.052; 4.0 -0.202 (SE .23) / +0.124 / -0.078; 4.5 -0.151 / +0.170 / +0.018; off -0.144 / +0.197 / +0.053. DC40all 24 seeds: xMSE -0.251% (SE .18), bill +0.125%, xnet -0.127%, aggregate -0.080%. DC40 at L12-14 on DC35, 24 seeds: xnet -0.135% (SE .155), aggregate -0.089%. Store n6: xMSE +0.03 (pooled ~-0.07).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_kinkoracle_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_decomp_dc35_v1.py](code/cand_decomp_dc35_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4216](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4216)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4289](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4289)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4296](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4296)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4302](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4302)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4312](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4312)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4322](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4322)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4325](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4325)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4327](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4327)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4452](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4452)

<details>
<summary>All result files and records (5)</summary>

- [CONF24_dc40all_v1.json](CONF24_dc40all_v1.json) · result
- [CONF24_dc40_1214_v1.json](CONF24_dc40_1214_v1.json) · result
- [L2H_dc45all_v1.json](L2H_dc45all_v1.json) · result
- [fragment_dc35_from_u64_v1.py.txt](fragment_dc35_from_u64_v1.py.txt) · code
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #336570: join ordering and three cutoff changes; a public network failed at .430 s residual](../../../submissions/336570/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-dead-cut-z-raised-3-0902-3-5-0) · [Copy provenance](../../../PROVENANCE.json)

