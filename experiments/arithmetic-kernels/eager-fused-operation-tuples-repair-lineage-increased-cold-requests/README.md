# Eager fused-operation tuples on the repair lineage increased cold requests and backend time

2026-10-17T16:06:44Z · experiment · rejected · measured

Enabling eager fused-operation tuples preserved prediction bytes but did not yield a request reduction because the active forward and unwind paths did not consume those tuples. The variant was rejected.

Recorded measurement:

    Warm requests +169 / -54; backend +3.7% / +0.8%; cold net +617 requests and +11% backend.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_lvfeager_v27e.py](code/candidate_lvfeager_v27e.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4880](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4880)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [PREFIRE_v27e.txt](PREFIRE_v27e.txt) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-eager-fused-operation-tuples-repair-lineage-increased-cold-requests) · [Copy provenance](../../../PROVENANCE.json)

