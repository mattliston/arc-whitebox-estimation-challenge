# Birth-fix and learned-edge dials on submission #336412: neither improved the confirmed final error

2026-10-16T16:55:33Z · experiment · rejected · measured

Two zero-bill dial families were re-tuned on the estimator submitted as #336412 (S2tr) with 8 noise-free seeds: the birth-fix gamma (_BFIX_GAMMA) at 0 and 0.2 both lose, the quadratic vertex is 0.119 worth -0.05%, and the vertex itself read null on 24 seeds, so 0.1 stays; the learned covariance-channel edge weight (_GN_EDGE -0.127 to 0, the full learned edge) read +0.073%. The -0.25 arm (GNE25) was started but its result was not reported.

Recorded measurement:

    BFIX0 (gamma 0): xMSE +1.781% (SE .44, 0/8 lower); BFIX2 (0.2): +0.811% (SE .62, 2/8); vertex gamma 0.119 worth -0.05%. BFIX119 (5205125a) 24 seeds: xnet +0.034% (SE .104), aggregate +0.015%, 11/24 lower (S8 +0.105, A80 -0.001). GNE0: +0.073% (SE .16, 3/8).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_ladder2h_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_s2tr_l2h_bfix119_v1.py](code/cand_s2tr_l2h_bfix119_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4247](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4247)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4255](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4255)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4346](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4346)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4364](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4364)

<details>
<summary>All result files and records (5)</summary>

- [L2H_bfix0_v1.json](L2H_bfix0_v1.json) · result
- [L2H_bfix2_v1.json](L2H_bfix2_v1.json) · result
- [CONF24_bfix119_v1.json](CONF24_bfix119_v1.json) · result
- [L2H_gne0_v1.json](L2H_gne0_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-birth-fix-learned-edge-dials-submission-336412-neither-improved) · [Copy provenance](../../../PROVENANCE.json)

