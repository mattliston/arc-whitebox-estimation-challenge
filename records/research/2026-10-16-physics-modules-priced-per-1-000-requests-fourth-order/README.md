# Physics modules priced per 1,000 requests: fourth-order, radial and covariance modules cost 15-25x the tree-strip diets

2026-10-16T15:01:24Z · finding · informational · measured

To find request budget for the static residual gate, each physics module of the estimator submitted as #336412 (S2tr) was switched off and priced as accuracy loss per 1,000 requests saved on 8 noise-free seeds. The learned edge was the only module near the tree-strip diets; the local fourth-order term (K4L), the F4c birth term and the learned covariance channel are 15-25x dearer per request. So the five-full sandwich diet stays and tree strips are the fallback.

Recorded measurement:

    EDGE off: -567 ops, bill -0.544%, xMSE +0.763% (SE .60, agg +0.83%), xnet +0.219% -> +0.39% per 1,000 requests; K4L local off: -955 ops, bill -0.224%, xMSE +5.47% -> +5.5% / 1,000; F4c off: -1,539 ops, bill -0.374%, xMSE +7.61% -> +4.7% / 1,000; GN cov off: -535 ops, xMSE +4.28% -> +7.7% / 1,000; five-full +0.92% / 1,000, ubown/cluster ~+1.0%, r002 ~+0.8%.

## Programs

- [build_modsw_v1.py](build_modsw_v1.py)
- [grade_modsw_v1.py](grade_modsw_v1.py)
- [cand_modsw_k4l_v1.py](cand_modsw_k4l_v1.py)
- [cand_modsw_f4c_v1.py](cand_modsw_f4c_v1.py)
- [cand_modsw_gncov_v1.py](cand_modsw_gncov_v1.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_S2tr_v1.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4261](../../TEAM_LOG_2026-10-01_to_10-22.md#L4261)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4266](../../TEAM_LOG_2026-10-01_to_10-22.md#L4266)

<details>
<summary>All result files and records (2)</summary>

- [MODSW_GRADE_latest.json](MODSW_GRADE_latest.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-physics-modules-priced-per-1-000-requests-fourth-order) · [Copy provenance](../../../PROVENANCE.json)

