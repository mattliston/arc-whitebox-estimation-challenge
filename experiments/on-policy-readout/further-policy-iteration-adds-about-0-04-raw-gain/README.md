# A further on-policy iteration adds about 0.04% raw gain on Mac and the ridge override is ineffective

2026-10-21T17:38:03Z · experiment · rejected · measured

Captures from the submitted 705-column trajectory supported a further fit with small native benefit. The requested ridge override did not reach the nested solver, so the supposed alternative fit was byte-identical and its redundant run was stopped.

Recorded measurement:

    - 2026-10-21T17:38:03Z [fable-lead] ITER2 FIT (fit_iter2_v1, ridge .1 damp .5, TRAIN90 only, captures of #337565's own trajectory): training -0.357% (89/90), physical rms 0.0598 / max layer 0.118. Seal lead/runs/astra_oct20_bill_v1/on_policy_first128_head90_v1/iter2_v1/fit_r0p1_d0p5/full90_COEFFICIENTS_v1.json. Exporting onto cap3 and cap4 stacks (scale 1.0) and measuring; extgate_s15_small (size cap) VM32+fresh launched 17:37Z.
    - 2026-10-21T17:39:30Z [fable-lead] iter2 stacks exported (scale 1.0): iter2_s10_cap3 (on #337565's files) and iter2_s10_cap4 (sci 17821bc8, on the held cap4 files; teacher-mismatch allowed: cap changes wall only) + extgate_s15_small (sci 12d2857b): VM32+fresh12 launched for all three; Mac32s queued sequentially.
    - 2026-10-21T17:58:42Z [fable-lead] Mac32 iter2_s10_cap3 vs 49bd: all32 -0.142% (raw -0.505, bill +0.364), held25 -0.383% (raw -0.747) -> iteration 2 adds ~0.04% raw over the fired file on Mac (near zero). BUG: fit_iter2 ridge override did not reach the gate fitter's own solver instance (fg.fs.cf), so the r0.03 iter2 seal == r0.1 seal; iter2_r003_s10_cap3 is a duplicate; killing its VM run.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_iter2_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_iter2_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_iter2_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_iter2_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/client_capture_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/client_capture_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_extended_gate_rd_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_extended_gate_rd_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_first128_onpolicy_tensors_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_first128_onpolicy_tensors_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/run_group_capture_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/run_group_capture_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_extended_gate_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_extended_gate_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_extended_gate_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_extended_gate_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_support_once_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_support_once_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_blocks2_fold0_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_blocks2_fold0_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_blocks_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_blocks_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_anchors_once_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_anchors_once_v1.py)

<details>
<summary>Additional programs and dependencies (32)</summary>

- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/build_parent_sources_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/build_parent_sources_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_gate_fold0_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_gate_fold0_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/run_capture_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/run_capture_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/first128_onpolicy_hook_v2.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/first128_onpolicy_hook_v2.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_extended_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_extended_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_extended_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_extended_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/first128_onpolicy_hook_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/first128_onpolicy_hook_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_remaining_once_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_remaining_once_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_blocks3_fold0_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_blocks3_fold0_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_second_step_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_second_step_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_parent_selftest_once_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_parent_selftest_once_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/prepare_capture_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/prepare_capture_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/check_parent_sources_import_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/check_parent_sources_import_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/review_anchor_invoice_sums_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/review_anchor_invoice_sums_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_blocks_fold0_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_blocks_fold0_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/audit_capture_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/audit_capture_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_anchors_once_v2.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_anchors_once_v2.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_parent_build_once_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_parent_build_once_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_prepare_once_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_prepare_once_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/qualify_parent_sources_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/qualify_parent_sources_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_blocks_full90_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_blocks_full90_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/check_observer_synthetic_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/check_observer_synthetic_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/run_capture_pipeline_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/run_capture_pipeline_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_science_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_science_v1.py)
- [code/astra_oct20_bill_v1/truth_response95_v1/fragment_mark_complete_line_v1.py](code/astra_oct20_bill_v1/truth_response95_v1/fragment_mark_complete_line_v1.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/client_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/client_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_group_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_group_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7506](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7506)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7507](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7507)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7518](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7518)

<details>
<summary>All result files and records (52)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [OPS_ANCHORS_STAGE_CLEARANCE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/OPS_ANCHORS_STAGE_CLEARANCE_v1.json) · result
- [ROOT_SELFTEST_REVIEW_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ROOT_SELFTEST_REVIEW_v1.json) · result
- [ROOT_ANCHORS_REVIEW_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ROOT_ANCHORS_REVIEW_v1.json) · result
- [REMAINING_LAUNCH_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/REMAINING_LAUNCH_v1.json) · result
- [ROOT_ANCHORS_CUSTODY_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ROOT_ANCHORS_CUSTODY_v1.json) · result
- [BUILD_LAUNCH_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/BUILD_LAUNCH_v1.json) · result
- [INDEPENDENT_ANCHOR_INVOICE_SUM_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/INDEPENDENT_ANCHOR_INVOICE_SUM_v1.json) · result
- [PREPARE_LAUNCH_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/PREPARE_LAUNCH_v1.json) · result
- [PREPARE_FINAL_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/PREPARE_FINAL_v1.json) · result
- [ROOT_SECOND_FIT_REVIEW_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ROOT_SECOND_FIT_REVIEW_v1.json) · result
- [SUPPORT_2_FINAL_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SUPPORT_2_FINAL_v1.json) · result
- [OBSERVER_SOURCE_REVIEW_READY_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/OBSERVER_SOURCE_REVIEW_READY_v1.json) · result
- [REMAINING_SOURCE_STAGE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/REMAINING_SOURCE_STAGE_v1.json) · result
- [ANCHOR_SOURCE_STAGE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ANCHOR_SOURCE_STAGE_v1.json) · result
- [REMAINING_AND_FIT_STAGE_READY_v2.json](../gate-times-first-128-readout-0-148839615-cross-validation/REMAINING_AND_FIT_STAGE_READY_v2.json) · result
- [STATIC_SEMANTIC_RESOURCE_AUDIT_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/STATIC_SEMANTIC_RESOURCE_AUDIT_v1.json) · result
- [ROOT_REMAINING_REVIEW_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ROOT_REMAINING_REVIEW_v1.json) · result
- [ROOT_QUALIFY_REVIEW_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ROOT_QUALIFY_REVIEW_v1.json) · result
- [SUPPORT_1_FINAL_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SUPPORT_1_FINAL_v1.json) · result
- [ANCHORS_LAUNCH_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ANCHORS_LAUNCH_v1.json) · result
- [PHASE_A_SOURCE_RECIPE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/PHASE_A_SOURCE_RECIPE_v1.json) · result
- [PARENT_BUILD_RECIPE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/PARENT_BUILD_RECIPE_v1.json) · result
- [ROOT_BUILD_REVIEW_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ROOT_BUILD_REVIEW_v1.json) · result
- [SUPPORT_0_FINAL_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SUPPORT_0_FINAL_v1.json) · result
- [SUPPORT_SOURCE_STAGE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SUPPORT_SOURCE_STAGE_v1.json) · result
- [REMAINING_AND_FIT_STAGE_READY_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/REMAINING_AND_FIT_STAGE_READY_v1.json) · result
- [SUPPORT_SEQUENCE_FINAL_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SUPPORT_SEQUENCE_FINAL_v1.json) · result
- [SELFTEST_LAUNCH_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SELFTEST_LAUNCH_v1.json) · result
- [ROOT_SUPPORT_SEQUENCE_REVIEW_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ROOT_SUPPORT_SEQUENCE_REVIEW_v1.json) · result
- [PARENT_SOURCES_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/PARENT_SOURCES_v1.json) · result
- [PARENT_SELFTEST_FINAL_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/PARENT_SELFTEST_FINAL_v1.json) · result
- [SELFTEST_STAGE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SELFTEST_STAGE_v1.json) · result
- [ROOT_PREPARE_REVIEW_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ROOT_PREPARE_REVIEW_v1.json) · result
- [SUPPORT_2_LAUNCH_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SUPPORT_2_LAUNCH_v1.json) · result
- [ANCHOR_STAGE_READY_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ANCHOR_STAGE_READY_v1.json) · result
- [PARENT_SELFTEST_RECIPE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/PARENT_SELFTEST_RECIPE_v1.json) · result
- [OPS_REMAINING522_AND_FIXEDFIT_CLEARANCE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/OPS_REMAINING522_AND_FIXEDFIT_CLEARANCE_v1.json) · result
- [PARENT_BUILD_FINAL_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/PARENT_BUILD_FINAL_v1.json) · result
- [PREREG_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/PREREG_v1.json) · result
- [SUPPORT_0_LAUNCH_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SUPPORT_0_LAUNCH_v1.json) · result
- [SUPPORT_1_LAUNCH_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SUPPORT_1_LAUNCH_v1.json) · result
- [PARENT_QUALIFY_RECIPE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/PARENT_QUALIFY_RECIPE_v1.json) · result
- [OBSERVER_SELFTEST_RECIPE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/OBSERVER_SELFTEST_RECIPE_v1.json) · result
- [SECOND_FIT_RECIPE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SECOND_FIT_RECIPE_v1.json) · result
- [ANCHOR_CUSTODY_COLLECTED_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ANCHOR_CUSTODY_COLLECTED_v1.json) · result
- [ANCHORS_FINAL_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/ANCHORS_FINAL_v1.json) · result
- [PREDICT_MULTIPLICITY_AUDIT_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/PREDICT_MULTIPLICITY_AUDIT_v1.json) · result
- [SOURCE_SCOPE_AUDIT_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/SOURCE_SCOPE_AUDIT_v1.json) · result
- [FULL_EXPERIMENT_SOURCE_RECIPE_v1.json](../gate-times-first-128-readout-0-148839615-cross-validation/FULL_EXPERIMENT_SOURCE_RECIPE_v1.json) · result
- [RECIPE_ITER2_v1.json](RECIPE_ITER2_v1.json) · result
- [PLAN_ITER2_v1.json](PLAN_ITER2_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-further-policy-iteration-adds-about-0-04-raw-gain) · [Copy provenance](../../../PROVENANCE.json)

