# Read-prefix cuts priced out: exact prefix 1.25 sigma -0.013% bill, prefix z 1.75 -0.10%, prefix z 1.5 +3.06% MSE

2026-10-16T14:33:56Z · experiment · rejected · measured

Cheaper versions of the plane read were priced on seed 1 and killed when the cold bill could not reach the -0.15% bar: moving read columns from the exact prefix to the rank-2 tail (RT125L, 1.5 to 1.25 sigma at layers 7-14) saves almost nothing because the tail still pays the old-bank part of the read; the seven-level strip routing gain set to zero (HZ7G0) saves 0.011%; shortening the read-column prefix itself to z 1.75 saves 0.10%; and z 1.5 loses 3% of accuracy because the z 1.5-2 band is load-bearing. The read cost is per column read at all.

Recorded measurement:

    RT125L (b18c8378): bill -0.013%. HZ7G0 (ddc66d64, _HZ7_GAIN 0.03 -> 0): bill -0.011%. RC175 (728ebf86): bill -0.100% cold on seed 1. RC15 (6b20d7dc): xMSE +3.055% (SE .49, 0/8), bill -0.326%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_ladder2h_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_s2tr_l2h_rc15_v1.py](code/cand_s2tr_l2h_rc15_v1.py)
- [code/cand_s2tr_l2h_rc175_v1.py](code/cand_s2tr_l2h_rc175_v1.py)
- [code/cand_s2tr_l2h_rt125l_v1.py](code/cand_s2tr_l2h_rt125l_v1.py)
- [code/cand_s2tr_l2h_hz7g0_v1.py](code/cand_s2tr_l2h_hz7g0_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4209](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4209)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4225](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4225)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4227](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4227)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4241](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4241)

<details>
<summary>All result files and records (2)</summary>

- [L2H_rc15_v1.json](L2H_rc15_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-read-prefix-cuts-priced-out-exact-prefix-1-25) · [Copy provenance](../../../PROVENANCE.json)

