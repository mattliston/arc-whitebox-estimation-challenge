# Old-plane low-rank reads retain 3.3-4.3% error even with 320 exact columns and a rank-eight tail

2026-10-21T02:37:30Z · experiment · rejected · measured

Physical read diagnostics rejected whole and tail-only low-rank compression of old plane reads. The existing omitted-column diagonal-completion result also closed a proposed duplicate direction.

Recorded measurement:

    Old-Dv whole/tail low-rank closure is now evidenced on exact777a TRAIN35/150 physical reads: sampled rank128 leaves21.6-30.1% relative read error; oracle rank1285.0-14.4%; keeping320exact plusmean/rank8tail still3.3-4.3%. Coefficient-weighted coordinates barely help; no nativecandidate. D/breakthrough_v1/OLD_READ_CLOSURE_v1.json5b6429be and10proof receipts saved. Original diagnostic finish erroneously expectedL7-14 though actualL10-14; rank35 explicitly FAILED harness/guard, salvaged analysis-only after byte/schema/resource proof; rank150 rerunonly withimmutablev2 clean. Both entire predictions match777a. Proposed omitted-column JSKY diagonal completion was found EXACTLY closed by Oct18 six-join probe46-136xworse; no rerun. Breakthrough now authorized TRAIN4 originalCM16 birth-coefficient teacher capture and one sharedA64x4 anchoredfit with full1024 normalization, fixedLOO/no truth.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/breakthrough_v1/client_capture_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/client_capture_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/run_group_capture_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/run_group_capture_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/run_capture_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/run_capture_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/analyze_old_dv_v2.py](code/astra_oct20_bill_v1/breakthrough_v1/analyze_old_dv_v2.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/analyze_old_dv_v3.py](code/astra_oct20_bill_v1/breakthrough_v1/analyze_old_dv_v3.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/old_dv_hook_v2.py](code/astra_oct20_bill_v1/breakthrough_v1/old_dv_hook_v2.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/launch_analysis_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/launch_analysis_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/old_dv_hook_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/old_dv_hook_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/launch_tail_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/launch_tail_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/analyze_old_tail_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/analyze_old_tail_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/run_capture_v2.py](code/astra_oct20_bill_v1/breakthrough_v1/run_capture_v2.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/analyze_old_dv_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/analyze_old_dv_v1.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/astra_oct20_bill_v1/breakthrough_v1/fragment_mark_complete_line_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/fragment_mark_complete_line_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/client_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/client_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_group_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_group_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6468](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6468)

<details>
<summary>All result files and records (12)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [OLD_READ_CLOSURE_v1.json](OLD_READ_CLOSURE_v1.json) · result
- [STATUS_CURRENT_v1.json](STATUS_CURRENT_v1.json) · result
- [RECIPE_v2.json](RECIPE_v2.json) · result
- [PLAN_v2.json](PLAN_v2.json) · result
- [IDEAS_AND_HISTORY_AUDIT_v1.json](IDEAS_AND_HISTORY_AUDIT_v1.json) · result
- [OBSERVER_CHECK_v1.json](OBSERVER_CHECK_v1.json) · result
- [ACTIVE_HANDOFF_v1.md](ACTIVE_HANDOFF_v1.md) · record
- [OBSERVER_CHECK_v2.json](OBSERVER_CHECK_v2.json) · result
- [RECIPE_v1.json](RECIPE_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [W3_CURRENT_ENCODER_METRIC_SOURCE_AUDIT_v1.json](W3_CURRENT_ENCODER_METRIC_SOURCE_AUDIT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-old-plane-low-rank-reads-retain-3-3-4) · [Copy provenance](../../../PROVENANCE.json)

