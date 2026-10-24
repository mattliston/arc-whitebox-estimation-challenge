# Setup-only collection threshold and prediction threshold changes improved some gates but lacked full margin

2026-10-18T09:18:03Z · experiment · rejected · measured

Setup collection batching shortened calibrated startup, while larger prediction generation-0 thresholds traded runtime pauses against memory. A threshold of 14000 approached the memory cap and 10000 failed slow-setup residual margin, leading to a lower request cap.

Recorded measurement:

    Setup 4.61997 s → 4.43291 s; threshold 14000 peak RSS 7.479 GB; threshold 10000: normal median .342900, max .386773, wall 116.267; slow median .345988 > .345222.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct18_lead_v1/repair_gc10000_v1/candidate_restore13_gc10000_HELD_v1.py](code/astra_oct18_lead_v1/repair_gc10000_v1/candidate_restore13_gc10000_HELD_v1.py)
- [code/astra_oct18_bill_v1/setup_restore13_v1/candidate_union_setup_restore13_HELD_v1.py](code/astra_oct18_bill_v1/setup_restore13_v1/candidate_union_setup_restore13_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5479](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5479)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5506](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5506)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5511](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5511)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [RELEASE_EVIDENCE_GC10000_v1.json](RELEASE_EVIDENCE_GC10000_v1.json) · record
- [RELEASE_EVIDENCE_090B85F1_v1.json](RELEASE_EVIDENCE_090B85F1_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-setup-only-collection-threshold-prediction-threshold-changes-improved-some) · [Copy provenance](../../../PROVENANCE.json)

