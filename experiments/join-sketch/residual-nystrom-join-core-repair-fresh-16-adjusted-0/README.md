# Residual Nystrom join-core repair: fresh-16 adjusted -0.12%, out-of-sample-16 +0.44%

2026-10-14T20:47:12Z · experiment · rejected · measured

A residual Nystrom repair of the join core (a float64 Schur-complement residual added back to the sketched core) first stopped on a float32-only finiteness checker, then passed the fresh screen narrowly and failed the independent out-of-sample pool.

Recorded measurement:

    fresh 16: log net -0.1232807115877349% (geometric -0.12320475213614433%, SE 0.3190777754652266%, 8 lower; log MSE -0.3135287825028831%, log bill +0.19024807091514817%); OOS 16 +0.438557%, all 32 +0.157282%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_residual_nystrom_v1/build_residual_nystrom_v1.py](code/lead/runs/astra_residual_nystrom_v1/build_residual_nystrom_v1.py)
- [code/lead/runs/astra_residual_nystrom_v1/matched_c_residual_nystrom_v1/candidate.py](code/lead/runs/astra_residual_nystrom_v1/matched_c_residual_nystrom_v1/candidate.py)
- [code/lead/runs/astra_residual_nystrom_v1/residual_nystrom_fragment_v1.py](code/lead/runs/astra_residual_nystrom_v1/residual_nystrom_fragment_v1.py)
- [code/scout_native/k4liab_s51k2tw/candidate.py](code/scout_native/k4liab_s51k2tw/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2984](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2984)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2989](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2989)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3020](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3020)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3024](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3024)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3042](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3042)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3065](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3065)

<details>
<summary>All result files and records (4)</summary>

- [ASTRA_FRESH_RETIREMENT_v1.json](ASTRA_FRESH_RETIREMENT_v1.json) · record
- [ASTRA_OOS_RETIREMENT_v1.json](ASTRA_OOS_RETIREMENT_v1.json) · record
- [FINITE_VALIDATION_DELTA_PROOF_v2.json](FINITE_VALIDATION_DELTA_PROOF_v2.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-residual-nystrom-join-core-repair-fresh-16-adjusted-0) · [Copy provenance](../../../PROVENANCE.json)

