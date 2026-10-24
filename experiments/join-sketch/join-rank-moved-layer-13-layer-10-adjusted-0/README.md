# Join rank moved from layer 13 to layer 10: adjusted -0.073% with +400 requests

2026-10-16T01:42:52Z · experiment · rejected · measured

The 384/384/320 rank allocation was not cost-neutral because reducing the last join saved much less than increasing the middle join cost. The small uncertain benefit did not justify requests.

Recorded measurement:

    xMSE-1.254%,6/8;bill+1.182%;ops+400;xnet-.073%,SE.66

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_lead_screen_v1/grade_vs_2b108_opus_v1.py](code/lead/runs/opus_lead_screen_v1/grade_vs_2b108_opus_v1.py)
- [code/lead/runs/opus_lead_screen_v1/grade_vs_2b108_opus_v2.py](code/lead/runs/opus_lead_screen_v1/grade_vs_2b108_opus_v2.py)
- [code/lead/runs/opus_lead_screen_v1/regrade_noisefree_all_v1.py](code/lead/runs/opus_lead_screen_v1/regrade_noisefree_all_v1.py)
- [code/lead/runs/opus_lead_screen_v1/cand_r10_448_onF384_v1.py](code/lead/runs/opus_lead_screen_v1/cand_r10_448_onF384_v1.py)
- [code/lead/runs/opus_lead_screen_v1/cand_r13_448_onF384_v1.py](code/lead/runs/opus_lead_screen_v1/cand_r13_448_onF384_v1.py)
- [code/lead/runs/opus_lead_screen_v1/cand_r7_320_on2b108_v1.py](code/lead/runs/opus_lead_screen_v1/cand_r7_320_on2b108_v1.py)
- [code/lead/runs/opus_lead_screen_v1/cand_r7_320_r10_384_onF384_v1.py](code/lead/runs/opus_lead_screen_v1/cand_r7_320_r10_384_onF384_v1.py)
- [code/lead/runs/opus_lead_screen_v1/cand_r7_448_on2b108_v1.py](code/lead/runs/opus_lead_screen_v1/cand_r7_448_on2b108_v1.py)
- [code/lead/runs/opus_lead_screen_v1/cand_swap_r10_384_r13_320_onF384_v1.py](code/lead/runs/opus_lead_screen_v1/cand_swap_r10_384_r13_320_onF384_v1.py)
- [code/lead/runs/opus_lead_screen_v1/screen_judge_opus_v1.py](code/lead/runs/opus_lead_screen_v1/screen_judge_opus_v1.py)
- [code/lead/runs/opus_lead_screen_v1/screen_judge_opus_v2.py](code/lead/runs/opus_lead_screen_v1/screen_judge_opus_v2.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3651](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3651)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-join-rank-moved-layer-13-layer-10-adjusted-0) · [Copy provenance](../../../PROVENANCE.json)

