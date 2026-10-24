# Allocation-matched C profiler preserves prediction, invoice and free-trip counts

2026-10-21T08:47:02Z · tool · informational · measured

A revised profiler matched construction and start/stop behavior in control and observed arms. It preserved all required byte and invoice identities and isolated a diagnostic-only frozen-object census as a measurable hotspot.

Recorded measurement:

    C-profile v2 completed 08:18:53Z under root e7946b26 and ops clearance 74ba47d7. Controller221267 and both original guards retired clean; prediction bytes, 235862163509 FLOPs, 43581 requests, 4627 free trips, 42230 freed handles and operation invoice matched exactly. Independent custody audit `V/profile_first128_d399_v2/INDEPENDENT_AUDIT_v1.json` 18f7e42e rehashes22 files/15 recipe pins; no timing qualification from instrumentation. Actual measured source hotspot: `gc.get_freeze_count()` called solely to populate unused per-predict diagnostic `frozen_objects`, 8.581425ms C self. Required `gc.collect`, freeze, thresholds, wrapper identities and every numerical expression stay unchanged.
    Prior profile failurec135e244 preserved. The preallocated Python callback draft still added11tracked GC counts in metadata test, so it was not advanced. Standard C-profile initially added3 because control skipped observer start/stop method calls; matching botharms' state construction AND start/stop removed this discrepancy. `V/profile_first128_d399_v2/CPROFILE_METADATA_CHECK_py310_v1.json` SHA9039f6e4b922f1b732dfa49ce66de129b344e41497160e44d263de0c0f1d21f4 PASS:10,000steps, exact GC counts/collections/finalizers/freebatchsequence in noncyclic and cyclic fixtures (8youngcollections), unchanged GCthresholds, no SDKruntime/model/NumPy. Draftfailures/sources retained.

## Programs

- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/check_cprofile_metadata_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/check_cprofile_metadata_v1.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/run_diagnostic_v2.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/run_diagnostic_v2.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/audit_custody_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/audit_custody_v1.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_primed_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_primed_v1.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/run_group_profile_v2.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/run_group_profile_v2.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/client_profile_v2.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/client_profile_v2.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/check_metadata_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/check_metadata_v1.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_builtins_off_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_builtins_off_v1.py)
- [code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_unprimed_v1.py](code/astra_oct20_vm_frontier_v1/profile_first128_d399_v2/profile_candidate_cprofile_unprimed_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_bill_v1/profile_candidate_cprofile_v1.py](code/astra_oct20_bill_v1/profile_candidate_cprofile_v1.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/astra_oct20_bill_v1/profile_outside_dispatch_v2.py](code/astra_oct20_bill_v1/profile_outside_dispatch_v2.py)
- [code/astra_oct20_vm_frontier_v1/sync_exact4685_v1/sdk_snapshot_v1/_dispatch.py](code/astra_oct20_vm_frontier_v1/sync_exact4685_v1/sdk_snapshot_v1/_dispatch.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/server_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/server_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/client_linux_markline_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/client_linux_markline_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/run_group_linux_markline_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/run_group_linux_markline_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7045](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7045)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6947](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6947)

<details>
<summary>All result files and records (15)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [CPROFILE_METADATA_CHECK_py310_v1.json](CPROFILE_METADATA_CHECK_py310_v1.json) · result
- [ROOT_NATIVE_DIAGNOSTIC_REVIEW_v1.json](ROOT_NATIVE_DIAGNOSTIC_REVIEW_v1.json) · result
- [DRAFT_ALLOCATION_FAILURES_v1.json](DRAFT_ALLOCATION_FAILURES_v1.json) · result
- [PLAN_LINUX_v1.json](PLAN_LINUX_v1.json) · result
- [ROOT_SLOT_REVIEW_v1.json](ROOT_SLOT_REVIEW_v1.json) · result
- [PAID_FRONTIER_EXISTING_EVIDENCE_v1.json](PAID_FRONTIER_EXISTING_EVIDENCE_v1.json) · result
- [PAID_REPLACEMENT_BUDGET_CORRECTION_v1.json](PAID_REPLACEMENT_BUDGET_CORRECTION_v1.json) · result
- [LAUNCH_v1.json](LAUNCH_v1.json) · result
- [ROOT_STATIC_REVIEW_v1.json](ROOT_STATIC_REVIEW_v1.json) · result
- [INDEPENDENT_AUDIT_v1.json](INDEPENDENT_AUDIT_v1.json) · result
- [RECIPE_v1.json](RECIPE_v1.json) · result
- [STAGING_AND_LIMITATIONS_v1.json](STAGING_AND_LIMITATIONS_v1.json) · result
- [STATIC_REVIEW_v1.json](STATIC_REVIEW_v1.json) · result
- [HISTORICAL_TWO_STRIP_OWN_BILL_v1.json](HISTORICAL_TWO_STRIP_OWN_BILL_v1.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-10-21-allocation-matched-c-profiler-preserves-prediction-invoice-free-trip) · [Copy provenance](../../PROVENANCE.json)

