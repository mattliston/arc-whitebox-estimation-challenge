# Covariance sandwich at gate floor 0.01: final-layer error +7-16% for less than 1% fewer FLOPs

2026-10-10T03:05Z · experiment · rejected · derived

Dropping units with gate below 0.01 from C_pre = W' C W drops the cut units' covariances with the live ones (1-2e-3 of C_pre), which the estimator feels at the 10% level; those cross covariances are not low rank (rank 32 keeps 97-99%), so a corrected cut nets about zero metered FLOPs.

Recorded measurement:

    network 651 +16.32%, 378 +6.72% final-layer error for bill -0.65% / -0.75%; rank-8 correction holds 63-82% of the dropped energy; corrected cut net 0 to -0.06% of bill

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a44_cov`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_dcv_v1.py](code/fragment_dcv_v1.py)
- [code/probe_deadcov.py](code/probe_deadcov.py)
- [code/probe_deadcov_v2.py](code/probe_deadcov_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L495](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L495)
- [TEAM_LOG_2026-10-01_to_10-22.md#L502](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L502)
- [TEAM_LOG_2026-10-01_to_10-22.md#L506](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L506)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Padded covariance cut at gate floor 1e-3 adopted: -0.370% FLOPs on ten fresh networks](../../../records/research/2026-10-10-padded-covariance-cut-gate-floor-1e-3-adopted-0/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-covariance-sandwich-gate-floor-0-01-final-layer-error) · [Copy provenance](../../../PROVENANCE.json)

