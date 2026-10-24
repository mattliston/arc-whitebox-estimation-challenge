# Late-only 64-coordinate read cut at layers 13-14: two-network adjusted +1.78/+1.97%

2026-10-16T00:43:46Z · experiment · rejected · measured

The late-only cut worsened both tested networks. A warm cost census located the largest arithmetic families and showed earlier read-cut damage persisted to the final layer.

Recorded measurement:

    MSE+2.25/+2.47%;net+1.78/+1.97%;newbornpush32.5%,youngread26.3%,star9.7%,joinpush8.8%,capturepush8.0%,oldread7.55%,other7.1%;matmul79%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_hunt_readext_v1/census_judge_v1.py](code/lead/runs/opus_hunt_readext_v1/census_judge_v1.py)
- [code/lead/runs/opus_hunt_readext_v1/census_judge_v2.py](code/lead/runs/opus_hunt_readext_v1/census_judge_v2.py)
- [code/lead/runs/opus_hunt_readext_v1/candidate_hunt_VE_early320_v1.py](code/lead/runs/opus_hunt_readext_v1/candidate_hunt_VE_early320_v1.py)
- [code/lead/runs/opus_hunt_readext_v1/candidate_hunt_VK_common256_v1.py](code/lead/runs/opus_hunt_readext_v1/candidate_hunt_VK_common256_v1.py)
- [code/lead/runs/opus_hunt_readext_v1/candidate_hunt_VL192_late_v1.py](code/lead/runs/opus_hunt_readext_v1/candidate_hunt_VL192_late_v1.py)
- [code/lead/runs/opus_hunt_readext_v1/compare_v1.py](code/lead/runs/opus_hunt_readext_v1/compare_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3605](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3605)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [census_b709warm.json](../common-rank-256-every-old-read-two-network-adjusted/census_b709warm.json) · result
- [census_b709warmfull.json](../common-rank-256-every-old-read-two-network-adjusted/census_b709warmfull.json) · result
- [HUNT_CENSUS_LATECUT_v2.json](../common-rank-256-every-old-read-two-network-adjusted/HUNT_CENSUS_LATECUT_v2.json) · result
- [HUNT_PILOT_DECISION_v1.json](../common-rank-256-every-old-read-two-network-adjusted/HUNT_PILOT_DECISION_v1.json) · result
- [log_census_b709.txt](../common-rank-256-every-old-read-two-network-adjusted/log_census_b709.txt) · result
- [log_census_b709full.txt](../common-rank-256-every-old-read-two-network-adjusted/log_census_b709full.txt) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-late-only-64-coordinate-read-cut-layers-13-14) · [Copy provenance](../../../PROVENANCE.json)

