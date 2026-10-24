# Trace-only raw-second moment conservation at layers 1-14: two-network adjusted +0.74%/+0.53%, closed

2026-10-14T18:38:28Z · experiment · rejected · measured

Conserving only the trace of the raw second moment across the learned head update (a scalar correction, 143,388 extra FLOPs) made both fixed networks worse.

Recorded measurement:

    adjusted +0.736620% / +0.533925%, paired +0.635221%; extra bill 143,388 FLOPs each

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_raw_second_trace_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_trace_conservation_HELD_v1.py](code/fragment_trace_conservation_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2972](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2972)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2978](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2978)

<details>
<summary>All result files and records (2)</summary>

- [SOURCE_HYPOTHESIS_HELD_v1.json](SOURCE_HYPOTHESIS_HELD_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-trace-only-raw-second-moment-conservation-layers-1-14) · [Copy provenance](../../../PROVENANCE.json)

