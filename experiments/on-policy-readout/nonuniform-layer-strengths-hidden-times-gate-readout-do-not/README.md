# Nonuniform layer strengths for the hidden-times-gate readout do not improve on uniform 1.5

2026-10-21T19:51:17Z · experiment · rejected · measured

Increasing later-layer or earlier-layer correction strength worsened native score. Reducing later layers to strength 1.0 was effectively tied, so uniform 1.5 remained the submitted setting.

Recorded measurement:

    - 2026-10-21T19:51:17Z [fable-lead] STRENGTH BRACKETS (hg r.03 seal, VM32 vs 49bd; uniform x1.5 = -0.351/-0.422): late2 (rows 9-15 x2.0) -0.098/-0.213 WORSE; late1 (rows 9-15 x1.0) -0.367/-0.430 = same as x1.5 -> late rows flat below 1.5, early rows carry the gain. Launched early2late1 (rows 1-8 x2.0, 9-15 x1.0) and early2late15 VM32. hg_r0p03 profile vs fired file: requests +0.96% vs +0.86%, puts -0.02% vs -0.13% (same ops class, ~+0.1% wall).
    - 2026-10-21T20:23:43Z [fable-lead] STRENGTH BRACKET CLOSED (hg r.03 seal, VM32 vs 49bd; uniform x1.5 = -0.351/-0.422): early2late1 (rows 1-8 x2.0, 9-15 x1.0) -0.214/-0.241 WORSE; late1 -0.367/-0.430 same; late2 worse. Uniform x1.5 stays.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/on_policy_first128_head90_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/export_blocks_v1.py](code/export_blocks_v1.py)
- [code/native_blocks_hg_r0p03_v1/early2late1/candidate_first128_onpolicy_blocks_science_v1.py](code/native_blocks_hg_r0p03_v1/early2late1/candidate_first128_onpolicy_blocks_science_v1.py)
- [code/native_blocks_hg_r0p03_v1/early2late15/candidate_first128_onpolicy_blocks_science_v1.py](code/native_blocks_hg_r0p03_v1/early2late15/candidate_first128_onpolicy_blocks_science_v1.py)
- [code/native_blocks_hg_r0p03_v1/late1/candidate_first128_onpolicy_blocks_science_v1.py](code/native_blocks_hg_r0p03_v1/late1/candidate_first128_onpolicy_blocks_science_v1.py)
- [code/native_blocks_hg_r0p03_v1/late2/candidate_first128_onpolicy_blocks_science_v1.py](code/native_blocks_hg_r0p03_v1/late2/candidate_first128_onpolicy_blocks_science_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7533](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7533)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7542](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7542)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-nonuniform-layer-strengths-hidden-times-gate-readout-do-not) · [Copy provenance](../../../PROVENANCE.json)

