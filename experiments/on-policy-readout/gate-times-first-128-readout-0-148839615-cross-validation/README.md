# Gate times first-128 readout: -0.148839615% cross-validation misses the fixed -0.15% threshold

2026-10-21T09:36:15Z · experiment · rejected · measured

A fixed gate-times-first-128 correction improved 85 of 90 networks but missed the preregistered mean threshold. Physical and fold gates passed; no full-data refit or export followed.

Recorded measurement:

    G one fixed fit ended09:32:35Z (2:32:35a.m.). FITTED86e8d886: CV90−0.148839615%,85/90lower,SE0.009222843%,pooled−0.148466915%, all5negative/allTRAINphysical gatesPASS and exact full90 original-parent tensorsPASS. Mean misses frozen−0.15% floor. No full90refit/model/export/native/fire, no retune/retry or threshold exception. Exact local JSON/guard/log receipts G/fit_receipts_v1; G/CLOSURE_v1.json SHA8509aae3870c94fc79c6409fb2d200a4ce5dd89e348f18136172e842c0b26495. Original guard41653bef all events/swap0,whole group retired,410.6MB RSS/331.1MB cgroup,612.20s,minavailable146.632GB; ops informed sole small slot free. Recovery216924 transitioned cleanly to wave3 fold4/full90 controllers243519/243521; wave2 both87/87+29/29clean/audited. Next preparation only: D/on_policy_first128_head90_v1 PREREG42c4f80e/source-scope2d82090b for authentic source-specific first128 on-policy second step, no new jobs/capture authorized.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

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
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/build_parent_sources_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/build_parent_sources_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_gate_fold0_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_gate_fold0_v1.py)

<details>
<summary>Additional programs and dependencies (37)</summary>

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
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_iter2_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_iter2_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/check_parent_sources_import_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/check_parent_sources_import_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/review_anchor_invoice_sums_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/review_anchor_invoice_sums_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_blocks_fold0_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/lc_blocks_fold0_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/audit_capture_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/audit_capture_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_anchors_once_v2.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_anchors_once_v2.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_parent_build_once_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_parent_build_once_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_prepare_once_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/dispatch_prepare_once_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/qualify_parent_sources_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/qualify_parent_sources_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_blocks_full90_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/fit_blocks_full90_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_iter2_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/export_iter2_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/check_observer_synthetic_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/check_observer_synthetic_v1.py)
- [code/astra_oct20_bill_v1/on_policy_first128_head90_v1/run_capture_pipeline_v1.py](code/astra_oct20_bill_v1/on_policy_first128_head90_v1/run_capture_pipeline_v1.py)
- [code/astra_oct20_bill_v1/first128_gate90_v1/fit_first_gate_v1.py](code/astra_oct20_bill_v1/first128_gate90_v1/fit_first_gate_v1.py)
- [code/astra_oct20_bill_v1/first128_gate90_v1/price_first_gate_v1.py](code/astra_oct20_bill_v1/first128_gate90_v1/price_first_gate_v1.py)
- [code/astra_oct20_bill_v1/first128_gate90_v1/GATE_FIRST128_TAIL_v1.py](code/astra_oct20_bill_v1/first128_gate90_v1/GATE_FIRST128_TAIL_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_science_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_science_v1.py)
- [code/astra_oct20_bill_v1/truth_response95_v1/fragment_mark_complete_line_v1.py](code/astra_oct20_bill_v1/truth_response95_v1/fragment_mark_complete_line_v1.py)
- [code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py](code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/client_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/client_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_group_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_group_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7151](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7151)

<details>
<summary>All result files and records (53)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [OPS_ANCHORS_STAGE_CLEARANCE_v1.json](OPS_ANCHORS_STAGE_CLEARANCE_v1.json) · result
- [ROOT_SELFTEST_REVIEW_v1.json](ROOT_SELFTEST_REVIEW_v1.json) · result
- [ROOT_ANCHORS_REVIEW_v1.json](ROOT_ANCHORS_REVIEW_v1.json) · result
- [REMAINING_LAUNCH_v1.json](REMAINING_LAUNCH_v1.json) · result
- [ROOT_ANCHORS_CUSTODY_v1.json](ROOT_ANCHORS_CUSTODY_v1.json) · result
- [BUILD_LAUNCH_v1.json](BUILD_LAUNCH_v1.json) · result
- [INDEPENDENT_ANCHOR_INVOICE_SUM_v1.json](INDEPENDENT_ANCHOR_INVOICE_SUM_v1.json) · result
- [PREPARE_LAUNCH_v1.json](PREPARE_LAUNCH_v1.json) · result
- [PREPARE_FINAL_v1.json](PREPARE_FINAL_v1.json) · result
- [ROOT_SECOND_FIT_REVIEW_v1.json](ROOT_SECOND_FIT_REVIEW_v1.json) · result
- [SUPPORT_2_FINAL_v1.json](SUPPORT_2_FINAL_v1.json) · result
- [OBSERVER_SOURCE_REVIEW_READY_v1.json](OBSERVER_SOURCE_REVIEW_READY_v1.json) · result
- [REMAINING_SOURCE_STAGE_v1.json](REMAINING_SOURCE_STAGE_v1.json) · result
- [ANCHOR_SOURCE_STAGE_v1.json](ANCHOR_SOURCE_STAGE_v1.json) · result
- [REMAINING_AND_FIT_STAGE_READY_v2.json](REMAINING_AND_FIT_STAGE_READY_v2.json) · result
- [STATIC_SEMANTIC_RESOURCE_AUDIT_v1.json](STATIC_SEMANTIC_RESOURCE_AUDIT_v1.json) · result
- [ROOT_REMAINING_REVIEW_v1.json](ROOT_REMAINING_REVIEW_v1.json) · result
- [ROOT_QUALIFY_REVIEW_v1.json](ROOT_QUALIFY_REVIEW_v1.json) · result
- [SUPPORT_1_FINAL_v1.json](SUPPORT_1_FINAL_v1.json) · result
- [ANCHORS_LAUNCH_v1.json](ANCHORS_LAUNCH_v1.json) · result
- [PHASE_A_SOURCE_RECIPE_v1.json](PHASE_A_SOURCE_RECIPE_v1.json) · result
- [PARENT_BUILD_RECIPE_v1.json](PARENT_BUILD_RECIPE_v1.json) · result
- [ROOT_BUILD_REVIEW_v1.json](ROOT_BUILD_REVIEW_v1.json) · result
- [SUPPORT_0_FINAL_v1.json](SUPPORT_0_FINAL_v1.json) · result
- [SUPPORT_SOURCE_STAGE_v1.json](SUPPORT_SOURCE_STAGE_v1.json) · result
- [REMAINING_AND_FIT_STAGE_READY_v1.json](REMAINING_AND_FIT_STAGE_READY_v1.json) · result
- [SUPPORT_SEQUENCE_FINAL_v1.json](SUPPORT_SEQUENCE_FINAL_v1.json) · result
- [SELFTEST_LAUNCH_v1.json](SELFTEST_LAUNCH_v1.json) · result
- [ROOT_SUPPORT_SEQUENCE_REVIEW_v1.json](ROOT_SUPPORT_SEQUENCE_REVIEW_v1.json) · result
- [PARENT_SOURCES_v1.json](PARENT_SOURCES_v1.json) · result
- [PARENT_SELFTEST_FINAL_v1.json](PARENT_SELFTEST_FINAL_v1.json) · result
- [SELFTEST_STAGE_v1.json](SELFTEST_STAGE_v1.json) · result
- [ROOT_PREPARE_REVIEW_v1.json](ROOT_PREPARE_REVIEW_v1.json) · result
- [SUPPORT_2_LAUNCH_v1.json](SUPPORT_2_LAUNCH_v1.json) · result
- [ANCHOR_STAGE_READY_v1.json](ANCHOR_STAGE_READY_v1.json) · result
- [PARENT_SELFTEST_RECIPE_v1.json](PARENT_SELFTEST_RECIPE_v1.json) · result
- [OPS_REMAINING522_AND_FIXEDFIT_CLEARANCE_v1.json](OPS_REMAINING522_AND_FIXEDFIT_CLEARANCE_v1.json) · result
- [PARENT_BUILD_FINAL_v1.json](PARENT_BUILD_FINAL_v1.json) · result
- [PREREG_v1.json](PREREG_v1.json) · result
- [SUPPORT_0_LAUNCH_v1.json](SUPPORT_0_LAUNCH_v1.json) · result
- [SUPPORT_1_LAUNCH_v1.json](SUPPORT_1_LAUNCH_v1.json) · result
- [PARENT_QUALIFY_RECIPE_v1.json](PARENT_QUALIFY_RECIPE_v1.json) · result
- [OBSERVER_SELFTEST_RECIPE_v1.json](OBSERVER_SELFTEST_RECIPE_v1.json) · result
- [SECOND_FIT_RECIPE_v1.json](SECOND_FIT_RECIPE_v1.json) · result
- [ANCHOR_CUSTODY_COLLECTED_v1.json](ANCHOR_CUSTODY_COLLECTED_v1.json) · result
- [ANCHORS_FINAL_v1.json](ANCHORS_FINAL_v1.json) · result
- [PREDICT_MULTIPLICITY_AUDIT_v1.json](PREDICT_MULTIPLICITY_AUDIT_v1.json) · result
- [SOURCE_SCOPE_AUDIT_v1.json](SOURCE_SCOPE_AUDIT_v1.json) · result
- [FULL_EXPERIMENT_SOURCE_RECIPE_v1.json](FULL_EXPERIMENT_SOURCE_RECIPE_v1.json) · result
- [CLOSURE_v1.json](CLOSURE_v1.json) · result
- [PREREG_v1-bfb1f9a1.json](PREREG_v1-bfb1f9a1.json) · result
- [FITTED_v1.json](FITTED_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-gate-times-first-128-readout-0-148839615-cross-validation) · [Copy provenance](../../../PROVENANCE.json)

