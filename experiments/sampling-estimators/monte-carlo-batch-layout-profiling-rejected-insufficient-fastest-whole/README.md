# Monte Carlo batch/layout profiling rejected as insufficient: fastest whole-producer forecast 1.019x

2026-10-12T16:23:13Z · experiment · rejected · measured

The truth producer was profiled across five batch sizes and three matrix-product layouts without changing the running chain. The best matrix-product rate implied only a small overall improvement, so no producer change was adopted.

Recorded measurement:

    batch 32768 158,614 samples/s (5.32 TFLOPS); batch 8192 162,390 samples/s = 1.024x the matmul chain, about 1.019x the whole producer (the chain is 78.8% of a batch)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst/fresh`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/loader.py](code/loader.py)
- [code/profile_mc_v2.py](code/profile_mc_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1524](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1524)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [profile_mc_v2.out](profile_mc_v2.out) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-monte-carlo-batch-layout-profiling-rejected-insufficient-fastest-whole) · [Copy provenance](../../../PROVENANCE.json)

