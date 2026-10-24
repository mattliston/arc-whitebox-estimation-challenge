# Unused frozen-object census removed: exact predictions and FLOPs, 13 cold residuals all lower

2026-10-21T08:47:02Z · experiment · adopted · measured

Profiling identified 8.581425 ms in gc.get_freeze_count used only for an unused diagnostic field. Replacing that expression with None preserved required garbage collection, numeric work and invoices; all 13 cold residuals improved by -1.883641315% to -6.737026781%.

Recorded measurement:

    C-profile v2 completed 08:18:53Z under root e7946b26 and ops clearance 74ba47d7. Controller221267 and both original guards retired clean; prediction bytes, 235862163509 FLOPs, 43581 requests, 4627 free trips, 42230 freed handles and operation invoice matched exactly. Independent custody audit `V/profile_first128_d399_v2/INDEPENDENT_AUDIT_v1.json` 18f7e42e rehashes22 files/15 recipe pins; no timing qualification from instrumentation. Actual measured source hotspot: `gc.get_freeze_count()` called solely to populate unused per-predict diagnostic `frozen_objects`, 8.581425ms C self. Required `gc.collect`, freeze, thresholds, wrapper identities and every numerical expression stay unchanged.
    Root bf11496c approved only replacing that diagnostic expression with `None`. Four immutable source descendants live in `D/runtime_first128_frozen_census_v1` (H): paid SCI49bd2ba3/RELf9381d09, zero-sync SCI190064a9/RELe20559ec. Source AST and full NumPy-free imports passed; eight imports and eight metadata return/exception/finally fixtures, zero models. Guard225620 ran08:28:29–32Z, peak228151296B, child events0/swap0/fully retired. `READY_SOURCE_v1.json` c4cc409d pins import4bcedd9f/guard38e7b8db. Build description had8.580769ms typo; immutable erratum e5964669 records actual8.581425ms, no code effect.
    FinalSCI49bd/RELf938 cold13 finished09:11:26Z/session94986exit0, queue77740restored. All13actualpairedresiduals lower: mean−3.863232078%,range−6.737026781..−1.883641315%; allpredictionbyteanchors, original5coldinvoiceanchors, allactualinvoices verified. Resultd7952561, ROOT_RESULT_REVIEW2bdb7dd0;26originalguards,maxRSS7407108096B,minfree77%,swapgrowth0. This is cold-only and not hosted guarantee. Distinct warm8 original-first-prefix pairs are nowLIVE session65595, launched09:17:34Z/drained09:18:13Z. NC/root_warm8_v1 root785f0175/preflight66f5ff19, planfde6e774/isolatione837497c/runnercb1bfad2/analyzer40abc12a. Fixedpairs35→150,35→200,250→259,277→287,16→293,16→300,310→312,310→316; original125s/7.5GB/free50guards, samef938/660sources, nofailedpartialreuse. Third-position200/300/316nowsecond have reducedretaineddepth and actualnewinvoices; nooldbilltransfer. All8warmtargets fixed, prefixobservationsseparate. NootherMacnative/setup untilretired. Do notjudgepartialtimes.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_release_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_release_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/audit_cold13_warm8_counterbalanced_v2.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/audit_cold13_warm8_counterbalanced_v2.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/run_imports_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/run_imports_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_science_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_science_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/audit_cold13_warm8_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/audit_cold13_warm8_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_zero_sync_nocensus_science_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_zero_sync_nocensus_science_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/build_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/build_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/build_release_evidence_index_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/build_release_evidence_index_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/audit_native32_vm_custody_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/audit_native32_vm_custody_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/check_import_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/check_import_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_zero_sync_nocensus_release_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/candidate_first128_zero_sync_nocensus_release_v1.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/check_cprofile_metadata_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/check_cprofile_metadata_v1.py)

<details>
<summary>Additional programs and dependencies (21)</summary>

- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/run_diagnostic_v2.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/run_diagnostic_v2.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/audit_custody_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/audit_custody_v1.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_primed_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_primed_v1.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/run_group_profile_v2.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/run_group_profile_v2.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/client_profile_v2.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/client_profile_v2.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/check_metadata_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/check_metadata_v1.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_builtins_off_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_builtins_off_v1.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_unprimed_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_unprimed_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/root_warm8_v1/run_isolated_root_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/root_warm8_v1/run_isolated_root_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/root_warm8_v1/analyze_coverage_v1.py](code/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/root_warm8_v1/analyze_coverage_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_bill_v1/profile_candidate_cprofile_v1.py](code/astra_oct20_bill_v1/profile_candidate_cprofile_v1.py)
- [code/astra_oct20_bill_v1/profile_outside_dispatch_v2.py](code/astra_oct20_bill_v1/profile_outside_dispatch_v2.py)
- [code/astra_oct20_vm_frontier_v1/sync_exact4685_v1/sdk_snapshot_v1/_dispatch.py](code/astra_oct20_vm_frontier_v1/sync_exact4685_v1/sdk_snapshot_v1/_dispatch.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/server_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/server_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/client_linux_markline_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/client_linux_markline_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/run_group_linux_markline_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/run_group_linux_markline_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7045](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7045)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7047](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7047)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7130](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7130)

<details>
<summary>All result files and records (55)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [RELEASE_EVIDENCE_INDEX_v3.json](RELEASE_EVIDENCE_INDEX_v3.json) · result
- [BUILD_v1.json](BUILD_v1.json) · result
- [EVIDENCE_CORRECTION_v1.json](EVIDENCE_CORRECTION_v1.json) · result
- [RELEASE_EVIDENCE_INDEX_v2.json](RELEASE_EVIDENCE_INDEX_v2.json) · result
- [LIVE_STATUS_20261021T085927Z_v1.json](LIVE_STATUS_20261021T085927Z_v1.json) · result
- [VM_PACKET_DEPLOY_v1.json](VM_PACKET_DEPLOY_v1.json) · result
- [NATIVE_STAGE_COMPLETION_EVENTS_v1.json](NATIVE_STAGE_COMPLETION_EVENTS_v1.json) · result
- [ROOT_VM32_AND_KNOWNCOLD4_CUSTODY_v1.json](ROOT_VM32_AND_KNOWNCOLD4_CUSTODY_v1.json) · result
- [IMPORT_CHECK_v1.json](IMPORT_CHECK_v1.json) · result
- [ROOT_SOURCE_IMPORT_REVIEW_v1.json](ROOT_SOURCE_IMPORT_REVIEW_v1.json) · result
- [INDEPENDENT_COLD13_WARM8_AUDIT_v1.json](INDEPENDENT_COLD13_WARM8_AUDIT_v1.json) · result
- [ROOT_RELEASE_SCRIPT_PINS_v1.json](ROOT_RELEASE_SCRIPT_PINS_v1.json) · result
- [ROOT_RUNTIME_AND_THIRD_BANK_VERDICT_v1.json](ROOT_RUNTIME_AND_THIRD_BANK_VERDICT_v1.json) · result
- [NATIVE_STAGE_PREFLIGHT_v1.json](NATIVE_STAGE_PREFLIGHT_v1.json) · result
- [NATIVE_REVIEW_v1.json](NATIVE_REVIEW_v1.json) · result
- [MAC32_DESCRIPTIVE_RUNTIME_VS_FE8_466_v1.json](MAC32_DESCRIPTIVE_RUNTIME_VS_FE8_466_v1.json) · result
- [NATIVE_LAUNCH_v1.json](NATIVE_LAUNCH_v1.json) · result
- [INDEPENDENT_COUNTERBALANCED_RUNTIME_AUDIT_v2.json](INDEPENDENT_COUNTERBALANCED_RUNTIME_AUDIT_v2.json) · result
- [REMAINING_PROFILE_FRONTIER_v1.json](REMAINING_PROFILE_FRONTIER_v1.json) · result
- [READY_SOURCE_v1.json](READY_SOURCE_v1.json) · result
- [IMPORT_RECIPE_v1.json](IMPORT_RECIPE_v1.json) · result
- [STRICT_FULL32_SCORE_EXECUTION_v1.json](STRICT_FULL32_SCORE_EXECUTION_v1.json) · result
- [RELEASE_EVIDENCE_INDEX_v1.json](RELEASE_EVIDENCE_INDEX_v1.json) · result
- [SOURCE_REVIEW_v1.json](SOURCE_REVIEW_v1.json) · result
- [ROOT_VM_STAGE_REVIEW_v1.json](ROOT_VM_STAGE_REVIEW_v1.json) · result
- [IMPORT_LAUNCH_v1.json](IMPORT_LAUNCH_v1.json) · result
- [PARENT_RANK316_REVIEW_v1.json](PARENT_RANK316_REVIEW_v1.json) · result
- [NATIVE_PLAN_LINUX_v1.json](NATIVE_PLAN_LINUX_v1.json) · result
- [WARM_CENSUS_MECHANISM_AND_LIMITATION_v1.json](WARM_CENSUS_MECHANISM_AND_LIMITATION_v1.json) · result
- [IMPORT_GUARD_v1.json](IMPORT_GUARD_v1.json) · result
- [RELEASE_COMMANDS_PREPARED_v1.json](RELEASE_COMMANDS_PREPARED_v1.json) · result
- [ROOT_NATIVE_DIAGNOSTIC_REVIEW_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/ROOT_NATIVE_DIAGNOSTIC_REVIEW_v1.json) · result
- [DRAFT_ALLOCATION_FAILURES_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/DRAFT_ALLOCATION_FAILURES_v1.json) · result
- [PLAN_LINUX_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/PLAN_LINUX_v1.json) · result
- [ROOT_SLOT_REVIEW_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/ROOT_SLOT_REVIEW_v1.json) · result
- [PAID_FRONTIER_EXISTING_EVIDENCE_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/PAID_FRONTIER_EXISTING_EVIDENCE_v1.json) · result
- [PAID_REPLACEMENT_BUDGET_CORRECTION_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/PAID_REPLACEMENT_BUDGET_CORRECTION_v1.json) · result
- [CPROFILE_METADATA_CHECK_py310_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/CPROFILE_METADATA_CHECK_py310_v1.json) · result
- [LAUNCH_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/LAUNCH_v1.json) · result
- [ROOT_STATIC_REVIEW_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/ROOT_STATIC_REVIEW_v1.json) · result
- [INDEPENDENT_AUDIT_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/INDEPENDENT_AUDIT_v1.json) · result
- [RECIPE_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/RECIPE_v1.json) · result
- [STAGING_AND_LIMITATIONS_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/STAGING_AND_LIMITATIONS_v1.json) · result
- [STATIC_REVIEW_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/STATIC_REVIEW_v1.json) · result
- [HISTORICAL_TWO_STRIP_OWN_BILL_v1.json](../../../instruments/allocation-matched-c-profiler-preserves-prediction-invoice-free-trip/HISTORICAL_TWO_STRIP_OWN_BILL_v1.json) · result
- [COVERAGE_PLAN_v1.json](COVERAGE_PLAN_v1.json) · result
- [ISOLATION_v1.json](ISOLATION_v1.json) · result
- [ROOT_RESULT_REVIEW_v1.json](ROOT_RESULT_REVIEW_v1.json) · result
- [ROOT_REVIEW_v1.json](ROOT_REVIEW_v1.json) · result
- [STATIC_COMMAND_SCHEMA_v1.json](STATIC_COMMAND_SCHEMA_v1.json) · result
- [ISOLATED_PLAN_v1.json](ISOLATED_PLAN_v1.json) · result
- [COVERAGE_RESULT_v1.json](COVERAGE_RESULT_v1.json) · result
- [READY_ROOT_REVIEW_v1.json](READY_ROOT_REVIEW_v1.json) · result
- [ROOT_LIVE_PREFLIGHT_v1.json](ROOT_LIVE_PREFLIGHT_v1.json) · result

</details>

## Related work

- shipped in: [Submission #337506: First-128 readout with fully charged residual-time repair](../../../submissions/337506/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-unused-frozen-object-census-removed-exact-predictions-flops-13) · [Copy provenance](../../../PROVENANCE.json)

