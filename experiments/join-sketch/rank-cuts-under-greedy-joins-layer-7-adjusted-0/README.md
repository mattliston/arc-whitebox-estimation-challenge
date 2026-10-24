# Rank cuts under greedy joins: layer 7 adjusted +0.01%, layer 13 +0.43%

2026-10-16T03:20:36Z · experiment · rejected · measured

Reducing either retained join rank from 384 to 320 did not meet the pre-registered incremental gain criterion and also added warm requests.

Recorded measurement:

    L7xMSE+1.06%,SE.67,bill-1.05%,net+.01%,SE.69;L13MSE+.87%,SE.44,bill-.45%,net+.43%;warmrequests+172/+103

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_rankgb_v1/grade_rankgb_v1.py](code/lead/runs/opus_rankgb_v1/grade_rankgb_v1.py)
- [code/lead/runs/opus_rankgb_v1/cand_GB128_V1_r7_320_v1.py](code/lead/runs/opus_rankgb_v1/cand_GB128_V1_r7_320_v1.py)
- [code/lead/runs/opus_rankgb_v1/cand_GB128_V2_r13_320_v1.py](code/lead/runs/opus_rankgb_v1/cand_GB128_V2_r13_320_v1.py)
- [code/lead/runs/opus_rankgb_v1/run_screen_v1.py](code/lead/runs/opus_rankgb_v1/run_screen_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/opus_lead_screen_v1/screen_judge_opus_v1.py](code/lead/runs/opus_lead_screen_v1/screen_judge_opus_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3770](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3770)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [RANKGB_DECISION_v1.json](RANKGB_DECISION_v1.json) · result
- [jobs_screen_v1.json](jobs_screen_v1.json) · result
- [RANKGB_MANIFEST_v1.json](RANKGB_MANIFEST_v1.json) · result
- [RANKGB_SCREEN_v1.json](RANKGB_SCREEN_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-rank-cuts-under-greedy-joins-layer-7-adjusted-0) · [Copy provenance](../../../PROVENANCE.json)

