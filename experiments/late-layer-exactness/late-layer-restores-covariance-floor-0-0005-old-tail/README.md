# Late-layer restores: covariance floor 0.0005, old-tail negative side, column-cut restore, star rank 8 all null or worse

2026-10-19T15:59:30Z · experiment · rejected · measured

The cousins of the paying young-crop restore were priced at layers 12-15: the dead-unit covariance floor lowered to 0.0005 at the last two covariance pushes is null, the old-tail negative side at 3.0 at layers 13+14 is dead, restoring the read column cut (2.5) at layers 13+14 hurts raw despite dropping the residual 10-15% (co-adapted, like the global restore), and a factored star of rank 8 into layers 14/15 or 12-15 is null or dead. Of the late-layer restores only the young crop (layers 13+14) and the old-tail positive side (layer 14) pay.

Recorded measurement:

    dcv1314_05: raw -0.052% (se .07), bill +0.106% vs stk1; ottn1314_30: raw -0.063% (se .14), bill +0.246% (adjusted +0.184%); rc1314_25: raw +0.320% (se .16, 8/28 lower), bill +0.094% (adjusted +0.411%; held-out +0.247%), residual 0.274->0.250 / 0.268->0.230 / 0.274->0.233 on ranks 35/150/200; ss_late8: raw -0.105% (se .07, 13/28), bill +0.156% (held-out +0.006%); ss_1215_8 (19 ranks): raw -0.017%, bill +0.239% (adjusted +0.222%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_dcv1314_05_sci_v1.py](code/candidate_dcv1314_05_sci_v1.py)
- [code/candidate_ottn1314_30_sci_v1.py](code/candidate_ottn1314_30_sci_v1.py)
- [code/candidate_rc1314_25_sci_v1.py](code/candidate_rc1314_25_sci_v1.py)
- [code/candidate_ss_late8_sci_v1.py](code/candidate_ss_late8_sci_v1.py)
- [code/candidate_ss_1215_8_sci_v1.py](code/candidate_ss_1215_8_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5965](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5965)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5967](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5967)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5968](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5968)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5969](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5969)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5970](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5970)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5972](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5972)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5973](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5973)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5977](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5977)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-late-layer-restores-covariance-floor-0-0005-old-tail) · [Copy provenance](../../../PROVENANCE.json)

