# Learned correction damping after the rank-384 join: four-network MSE +0.556/+0.171%

2026-10-16T01:51:22Z · experiment · rejected · measured

Reducing correction strength at the affected layers did not improve the larger-rank estimator. The learned head remained at its calibrated strength.

Recorded measurement:

    x.8L10-12xMSE+.556%,SE.10,4/4worse;x.9L10-15+.171%

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L3658](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3658)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-learned-correction-damping-after-rank-384-join-four-network) · [Copy provenance](../../../PROVENANCE.json)

