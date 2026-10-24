# Layer-7 join rank 448: four-network adjusted +0.74% despite MSE -0.91%

2026-10-16T01:30:51Z · experiment · rejected · measured

Increasing the first join rank improved all four raw-error measurements but cost more FLOPs and requests than it gained.

Recorded measurement:

    xMSE-.91%,4/4;bill+1.65%;ops+414..624;xnet+.74%

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L3636](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3636)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-layer-7-join-rank-448-four-network-adjusted-0) · [Copy provenance](../../../PROVENANCE.json)

