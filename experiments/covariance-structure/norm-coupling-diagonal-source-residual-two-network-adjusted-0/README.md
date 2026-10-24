# Norm-coupling diagonal-source residual: two-network adjusted +0.79%/-0.27%, geometric +0.26%

2026-10-14T19:43:26Z · experiment · rejected · measured

A residual term coupling the norm of the carried state to the diagonal source (fourteen source/consumer updates per network, trace balance below 1.75e-15) was mixed on the two fixed networks and worse on average.

Recorded measurement:

    adjusted 7100136 +0.787551%, 7100137 -0.268866%, geometric +0.257951%; mean log MSE +0.219806%, bill +0.037813% (~99.77M extra FLOPs)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_norm_coupling_residual_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/norm_coupling_residual_fragment_HELD_v1.py](code/norm_coupling_residual_fragment_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3030](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3030)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3033](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3033)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3034](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3034)

<details>
<summary>All result files and records (3)</summary>

- [ROOT_RETIREMENT_v1.json](ROOT_RETIREMENT_v1.json) · record
- [SOURCE_ALGEBRA_v1.json](SOURCE_ALGEBRA_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-norm-coupling-diagonal-source-residual-two-network-adjusted-0) · [Copy provenance](../../../PROVENANCE.json)

