# Heavy-column sketch squeezed to 1536/1024/1024: +0.04% adjusted versus the submitted plan

2026-10-18T03:10:55Z · experiment · rejected · measured

Reducing the layer-7, layer-10 and layer-13 heavy-column counts below the submitted plan saved additional bill but lost slightly more raw accuracy. The sketch reduction had saturated at 2048/1536/1536.

Recorded measurement:

    7 ranks: adjusted -1.80% versus control, +0.04% versus 2048/1536/1536; bill -0.30%, raw +0.34%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_v63j.py](code/candidate_v63j.py)
- [code/grade_store_v1.py](code/grade_store_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5105](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5105)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [GRADE_STORE_v1.json](GRADE_STORE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-heavy-column-sketch-squeezed-1536-1024-1024-0-04) · [Copy provenance](../../../PROVENANCE.json)

