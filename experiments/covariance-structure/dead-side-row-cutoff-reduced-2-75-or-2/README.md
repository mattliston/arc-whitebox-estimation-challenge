# Dead-side row cutoff reduced to 2.75 or 2.5: adjusted +0.154% and +1.255%

2026-10-16T08:24:07Z · experiment · rejected · measured

More aggressive dead-side cuts saved little because retained row counts were rounded to 64 and lost more accuracy than they saved.

Recorded measurement:

    2.75bill-.199%,MSE+.352%,SE.37,net+.154%;2.5bill-.232%,MSE+1.487%,SE1.6,net+1.255%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_deadcut_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_S2tr_dz25_v1.py](code/cand_S2tr_dz25_v1.py)
- [code/cand_S2tr_dz275_v1.py](code/cand_S2tr_dz275_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4038](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4038)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-dead-side-row-cutoff-reduced-2-75-or-2) · [Copy provenance](../../../PROVENANCE.json)

