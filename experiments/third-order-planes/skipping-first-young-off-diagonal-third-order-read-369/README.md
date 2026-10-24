# Skipping the first young off-diagonal third-order read: +369% raw error for -1.0% FLOPs

2026-10-19T03:39:25Z · experiment · rejected · measured

Skipping the read of the young off-diagonal CP plane at layer 1 (diagonal k3 kept) multiplied the final error: raw +369% for a 1.0% bill saving, so the earliest third-order plane content is essential to the closure; the layer-2 and layer-3 variants were killed unrun.

Recorded measurement:

    lt2 2f51a1d1 (7 ranks vs 95c): raw +369% (se 48, 0/7 lower), bill -1.0%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_early_plane_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_jsym_planeskip_lt2_v1.py](code/candidate_jsym_planeskip_lt2_v1.py)
- [code/run_judges_v1.sh](code/run_judges_v1.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5899](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5899)

<details>
<summary>All result files and records (2)</summary>

- [candidate_jsym_planeskip_lt2_v1_science7.log](candidate_jsym_planeskip_lt2_v1_science7.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-skipping-first-young-off-diagonal-third-order-read-369) · [Copy provenance](../../../PROVENANCE.json)

