# First-128 on-policy refit strength: zero added FLOPs and about -0.3% adjusted across native cohorts

2026-10-21T14:33:14Z · experiment · adopted · measured

Tensor-only second-step readout fitting improved native score without adding operations. Strengths from 0.5 to 2.0 were measured on the fixed Mac and Linux networks and the never-fit twelve-network confirmation set; the 1.5-strength fresh result retained two large regressions.

Recorded measurement:

    - 2026-10-21T14:33:14Z [fable-lead] FRESH12 confirmation of on-policy first128 second-step s1.0 (sci 6e8e9100) vs deployed 49bd on 12 never-fit cold singletons (seeds 7109001-12): adjusted -0.2998% (SE 0.1505), 8/12 lower, raw -0.2998%, bill +0.0000% (tensor-only). original8 -0.243% (5/8), extension4 -0.413% (3/4). Files lead/runs/fable_oct21_final_v1/fresh12_onpolicy_v1/results/. Proxy was -0.637%; transfer ~0.47. Mac32 and VM32 pending; strength 0.5/1.5/2.0 variants queued on VM.
    - 2026-10-21T14:34:57Z [fable-lead] MAC32 paired on-policy s1.0 (sci 6e8e9100 / rel 8185cf0b) vs deployed 49bd Mac32: all32 -0.3439% (SE 0.1398, 22/32), held25 -0.4648% (SE 0.1521, 18/25), seven +0.0878% (4/7), bill +0.0001%. Run lead/runs/astra_oct20_bill_v1/on_policy_first128_head90_v1/science32_first128_onpolicy_second90_parallel5_v1; compare lead/runs/fable_oct21_final_v1/MAC32_onpolicy_s1_vs_49bd_v1.json. Decision: release gates started for tag fable_onpolicy_second90_s1_8185cf0b_v1 (runtime byte-identical to #337506); fire if gates clear.
    - 2026-10-21T14:53:40Z [fable-lead] VM32 paired vs 49bd: on-policy s1.0 all32 -0.164% (SE 0.141, 20/32) held25 -0.205%; s1.5 all32 -0.385% (SE 0.164, 21/32) held25 -0.436%; bill 0. Mac32 s1.5 and fresh12 s1.5 launched 14:52-14:53Z; s2.0 VM32 queued. Wall numbers under VM load are confounded (identical-op file read +10%); requests/puts are the wall proxies; SM40-off probe (Smirnov young reads, _ASTRA_SM40_ON) queued paired with a reference. Extended fit (hidden2^2 + first128^2 features, 577 cols) running on VM, fold1 at 14:55Z.
    - 2026-10-21T15:00:56Z [fable-lead] STRENGTH CURVE vs 49bd: s1.0 Mac32 -0.344/-0.465 held, VM32 -0.164/-0.205, fresh12 -0.300 (8/12). s1.5 (sha aaec3e8e) Mac32 -0.431% (SE 0.190, 18/32) / held25 -0.527% (15/25), VM32 -0.385/-0.436, fresh12 -0.073% (SE 0.258, 9/12; two outliers +1.32/+1.85 on seeds 7109005/7109009). s2.0 VM32 scored pending. Mac runs science32_onpolicy_s1p5_parallel5_v1; fresh FRESH_*_s1p5_vs_49bd_v1.json on VM. #337534 grading at 43/100: residual max now 0.381 s (337506: 0.364) -> residual margin also thin today.
    - 2026-10-21T15:04:39Z [fable-lead] fresh12 s2.0 (sha c4b6087f) vs 49bd: -0.270% (SE 0.229, 10/12); VM32 s2.0 -0.290/-0.354, s0.5 -0.279/-0.317. Strength curve flat within noise; refit worth ~-0.3% +- 0.1 at s in [1,2].
    - 2026-10-21T15:16:14Z [fable-lead] Mac32 s2.0 (sha c4b6087f) vs 49bd: all32 -0.414% (SE 0.220, 19/32). SMX-off paired (12 ranks, same load): requests +0.2%, puts +0.8%, flops +0.65%, wall +11.6%, residual +13% -> CLOSED (Smirnov family is a wall optimization here). Platform scan: 83 recent submissions with rt max >= 100 s, 13 with TIME_EXHAUSTED (16%).
    Sweep result detail: - 2026-10-21T14:51:48Z | first128_onpolicy_second90_native32_v1 | on-policy first128 second-step refit s1.0 (#337534 file, sha 6e8e9100) | all32 adj -0.164% (se 0.141) | held25 adj -0.205% (se 0.166) | wins 20/32 | raw -0.162% | bill -0.002% | wall +9.91% | requests +0.00% | puts +0.00% | max-rank wall +18.83% | slow10 wall +3.33% req +0.00% adj -0.487% | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/first128_onpolicy_second90_native32_v1 | sha256 n/a | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_first128_onpolicy_second90.py
    - 2026-10-21T14:51:51Z | onpolicy_s1p5_native32_v1 | on-policy first128 second-step refit s1.5 | all32 adj -0.385% (se 0.164) | held25 adj -0.436% (se 0.192) | wins 21/32 | raw -0.385% | bill -0.000% | wall +10.22% | requests +0.00% | puts +0.00% | max-rank wall +18.66% | slow10 wall +15.93% req +0.00% adj -0.754% | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/onpolicy_s1p5_native32_v1 | sha256 n/a | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_onpolicy_s1p5.py
    - 2026-10-21T15:01:29Z | onpolicy_s2p0_native32_v1 | on-policy first128 second-step refit s2.0 | all32 adj -0.290% (se 0.231) | held25 adj -0.354% (se 0.283) | wins 19/32 | raw -0.288% | bill -0.002% | wall +25.63% | requests +0.00% | puts +0.00% | max-rank wall +52.41% | slow10 wall +7.14% req +0.00% adj -0.726% | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/onpolicy_s2p0_native32_v1 | sha256 n/a | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_onpolicy_s2p0.py
    - 2026-10-21T15:03:06Z | onpolicy_s0p5_native32_v1 | on-policy first128 second-step refit s0.5 | all32 adj -0.279% (se 0.113) | held25 adj -0.317% (se 0.138) | wins 22/32 | raw -0.279% | bill -0.000% | wall +29.51% | requests +0.00% | puts +0.00% | max-rank wall +44.38% | slow10 wall +41.99% req +0.00% adj -0.562% | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/onpolicy_s0p5_native32_v1 | sha256 n/a | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_onpolicy_s0p5.py

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
<summary>Additional programs and dependencies (32)</summary>

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L7423](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7423)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7424](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7424)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7427](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7427)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7430](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7430)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7440](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7440)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7443](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7443)
- [VM_SWEEP_RESULTS_2026-10-21.md#L11](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L11)
- [VM_SWEEP_RESULTS_2026-10-21.md#L12](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L12)
- [VM_SWEEP_RESULTS_2026-10-21.md#L15](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L15)
- [VM_SWEEP_RESULTS_2026-10-21.md#L16](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L16)

<details>
<summary>All result files and records (62)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [MAC32_onpolicy_s1_vs_49bd_v1.json](MAC32_onpolicy_s1_vs_49bd_v1.json) · result
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
- [CURRENT_SOURCE7_v1.json](CURRENT_SOURCE7_v1.json) · result
- [VS_BEST_GRADED_v1.json](VS_BEST_GRADED_v1.json) · result
- [PARALLEL32_COMPLETE_v1.json](PARALLEL32_COMPLETE_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [RESULT32_v1.json](RESULT32_v1.json) · result
- [CURRENT_SOURCE7_v1-bdb1224d.json](CURRENT_SOURCE7_v1-bdb1224d.json) · result
- [VS_BEST_GRADED_v1-35b7bbd6.json](VS_BEST_GRADED_v1-35b7bbd6.json) · result
- [PARALLEL32_COMPLETE_v1-59bfdb84.json](PARALLEL32_COMPLETE_v1-59bfdb84.json) · result
- [PLAN_v1-7a753a1e.json](PLAN_v1-7a753a1e.json) · result
- [RESULT32_v1-3f94c008.json](RESULT32_v1-3f94c008.json) · result
- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

## Related work

- shipped in: [Submission #337534: Tensor-only on-policy second readout fit; two hosted wall failures](../../../submissions/337534/README.md)
- shipped in: [Submission #337565: 705-column gate readout at strength 1.5 and grouped-read cap 3](../../../submissions/337565/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-first-128-policy-refit-strength-zero-added-flops-about) · [Copy provenance](../../../PROVENANCE.json)

