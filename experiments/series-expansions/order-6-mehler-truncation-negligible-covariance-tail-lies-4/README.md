# Order-6 Mehler truncation is negligible: its covariance tail lies 4-5 orders below the variance error

2026-10-11T04:05Z · experiment · rejected · derived

Pre-activation |rho| p99 stays below 0.3 (no pair above 0.7 at any layer), so the order 7-60 tail is 1e-10..2.5e-7 of the off-diagonal post covariance, 4-5 orders below the variance error; covariance error is cumulant structure, as the team had it.

Recorded measurement:

    |rho| p99 0.08 (L0) -> 0.29 (L15); tail moves the next layer's variance by rms 5e-12..2.5e-8 relative vs the 0.09-0.18% v error

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead_mehler_tail_v1.py](code/lead_mehler_tail_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L999](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L999)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1795](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1795)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [lead_mehler_tail_v1_7100136.out](lead_mehler_tail_v1_7100136.out) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-order-6-mehler-truncation-negligible-covariance-tail-lies-4) · [Copy provenance](../../../PROVENANCE.json)

