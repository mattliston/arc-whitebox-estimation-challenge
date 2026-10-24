# Classical float32 and float64-accumulated matrix products do not improve final accuracy

2026-10-16T08:38:49Z · experiment · rejected · measured

Turning off all fast matrix algorithms increased bill sharply and ran faster locally, but did not reduce error. Float64 accumulation of the same large products was nearly indistinguishable from classical float32.

Recorded measurement:

    95.7%FLOPsclassical;bill+68.2%;localwall~2xfaster;float32xMSE+.127%,SE.070,2/8;float64+.128%,SE.069;float64vsfloat32+.001%,SE.002

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_roundx_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_s2tr_rx_a0_v1.py](code/cand_s2tr_rx_a0_v1.py)
- [code/cand_s2tr_rx_b0_v1.py](code/cand_s2tr_rx_b0_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4051](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4051)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4050](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4050)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [guard_census_a0.json](guard_census_a0.json) · result
- [jobs_a0_v1.json](jobs_a0_v1.json) · result
- [jobs_b0_v1.json](jobs_b0_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-classical-float32-float64-accumulated-matrix-products-do-not-improve) · [Copy provenance](../../../PROVENANCE.json)

