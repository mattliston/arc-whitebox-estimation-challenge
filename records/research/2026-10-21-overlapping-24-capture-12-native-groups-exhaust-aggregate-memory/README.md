# Overlapping 24 capture and 12 native groups exhaust aggregate memory despite per-group limits

2026-10-21T06:59:42Z · incident · failed · measured

The shared slot count did not bound the simultaneous actual memory peak. Original reserve guards retired the work without OOM or swap; partial predictions were excluded and subsequent recovery separated or explicitly bounded concurrent stages.

Recorded measurement:

    VM incident: overlapping24onpolicy+up to12repaired-native groups exhausted aggregate memory despite original shared36 admissions. Parent memory.events.max0->128, zero OOM/kill/swap. Original19GB reserve guard retired native work06:53Z; no limits weakened, no jobs blindly killed. Repaired12 groups FAILED, no qualification; six onpolicy remaining87 controllers FAILED, all partial predictions inadmissible. Exact original failures preserved: D/runtime_smoff_npxoff_first128_v1/SHARED_MEMORY_FAILURE_REVIEW_v1.json b218d01e, OP/remaining_failure_v1/FAILURE_SUMMARY_v1.json e03e32a3. Parent controllers194848/196055/189185 and all6remaining confirmed absent06:56:53Z. Only originaltruths continue; root directSSH06:56:30Z confirmed172324MiBavailable, cgroup9.69GB, zeroSwap/OOM. HealthCPU84% after retirement (not assumed100%). Lead responsibility: dispatch overlap exceeded measured aggregate RAM; restarting requires reviewed separated schedule/new outputs, all original caps unchanged. Ops prepares at most12totalnative alongside12truth; no native retry authorized yet. One BLAS1 first-squarefit and runtime tiny/import may proceed after measured clearance.

## Programs

- [audit_native32_vm_custody_separated_v2.py](audit_native32_vm_custody_separated_v2.py)
- [build_v1.py](build_v1.py)
- [candidate_first128_smoff_npxoff_release_v1.py](candidate_first128_smoff_npxoff_release_v1.py)
- [audit_native32_vm_custody_v1.py](audit_native32_vm_custody_v1.py)
- [candidate_first128_smoff_npxoff_science_v1.py](candidate_first128_smoff_npxoff_science_v1.py)
- [check_import_v1.py](check_import_v1.py)
- [../../../experiments/fourth-order-terms/fourth-order-shared-polynomial-reuse-age-four-term-promising/code/astra_oct20_bill_v1/tail_npx_copy_exact_v1.py](../../../experiments/fourth-order-terms/fourth-order-shared-polynomial-reuse-age-four-term-promising/code/astra_oct20_bill_v1/tail_npx_copy_exact_v1.py)
- [../../../experiments/wall-and-residual/feature-pair-grouping-plus-kernel-scratch-compaction-reduce-requests/code/astra_oct20_bill_v1/tail_sm_direct_v1.py](../../../experiments/wall-and-residual/feature-pair-grouping-plus-kernel-scratch-compaction-reduce-requests/code/astra_oct20_bill_v1/tail_sm_direct_v1.py)
- [../../../experiments/wall-and-residual/disabling-direct-scratch-kernel-trades-125809-flops-2-623392/code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/tail_smoff_v1.py](../../../experiments/wall-and-residual/disabling-direct-scratch-kernel-trades-125809-flops-2-623392/code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/tail_smoff_v1.py)
- [../../../experiments/wall-and-residual/disabling-direct-scratch-kernel-trades-125809-flops-2-623392/code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/tail_npxoff_v1.py](../../../experiments/wall-and-residual/disabling-direct-scratch-kernel-trades-125809-flops-2-623392/code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/tail_npxoff_v1.py)
- [../2026-10-20-false-initial-rescue-flag-did-not-disable-residual-triggered/tail_disable_fast_rescue_astra_v1.py](../2026-10-20-false-initial-rescue-flag-did-not-disable-residual-triggered/tail_disable_fast_rescue_astra_v1.py)
- [../../../experiments/learned-correction/current-message-edge-context-refit-current-consumer-projector-metric/code/candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py](../../../experiments/learned-correction/current-message-edge-context-refit-current-consumer-projector-metric/code/candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6787](../../TEAM_LOG_2026-10-01_to_10-22.md#L6787)

<details>
<summary>All result files and records (24)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SHARED_MEMORY_FAILURE_REVIEW_v1.json](SHARED_MEMORY_FAILURE_REVIEW_v1.json) · result
- [FAILURE_SUMMARY_v1.json](FAILURE_SUMMARY_v1.json) · result
- [CAPTURE_FINAL_v1.json](CAPTURE_FINAL_v1.json) · result
- [PIPELINE_LAUNCH_v1.json](PIPELINE_LAUNCH_v1.json) · result
- [ROOT_CAPTURE_AND_FIT_REVIEW_v1.json](ROOT_CAPTURE_AND_FIT_REVIEW_v1.json) · result
- [BUILD_v1.json](BUILD_v1.json) · result
- [NATIVE_STAGE_COMPLETION_EVENTS_v1.json](NATIVE_STAGE_COMPLETION_EVENTS_v1.json) · result
- [STRICT_WATCH_LAUNCH_separated_v2.json](STRICT_WATCH_LAUNCH_separated_v2.json) · result
- [STRICT_WATCH_INPUT_separated_v2.json](STRICT_WATCH_INPUT_separated_v2.json) · result
- [NATIVE_REVIEW_v1.json](NATIVE_REVIEW_v1.json) · result
- [SCORING_PLAN_COMPATIBILITY_v1.json](SCORING_PLAN_COMPATIBILITY_v1.json) · result
- [ROOT_SEPARATED_RECOVERY_REVIEW_v1.json](ROOT_SEPARATED_RECOVERY_REVIEW_v1.json) · result
- [NATIVE_METADATA_COPY_v1.json](NATIVE_METADATA_COPY_v1.json) · result
- [SCORING_PLAN_LINUX_v1.json](SCORING_PLAN_LINUX_v1.json) · result
- [PREFLIGHT_FIRST_SQUARE_FIT_GUARD_v1.json](PREFLIGHT_FIRST_SQUARE_FIT_GUARD_v1.json) · result
- [STATIC_RECOVERY_SCHEDULE_REVIEW_v1.json](STATIC_RECOVERY_SCHEDULE_REVIEW_v1.json) · result
- [ROOT_SOURCE_REVIEW_v1.json](ROOT_SOURCE_REVIEW_v1.json) · result
- [READY_SOURCE_v1.json](READY_SOURCE_v1.json) · result
- [SEPARATED_BASELINE_CLARIFICATION_v1.json](SEPARATED_BASELINE_CLARIFICATION_v1.json) · result
- [ROOT_MAC32_REVIEW_v1.json](ROOT_MAC32_REVIEW_v1.json) · result
- [NATIVE_PLAN_LINUX_v1.json](NATIVE_PLAN_LINUX_v1.json) · result
- [LOCAL_REPORT_CUSTODY_CONTINUATION_v1.json](LOCAL_REPORT_CUSTODY_CONTINUATION_v1.json) · result
- [SEPARATED_PREFLIGHT_v2.json](SEPARATED_PREFLIGHT_v2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory) · [Copy provenance](../../../PROVENANCE.json)

