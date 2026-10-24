# Paid kernel and copy restorations preserve first-128 predictions and reduce three-network residual time 4.330961%

2026-10-21T07:35:57Z · experiment · adopted · measured

Restoring the grouped scratch kernel and copy behavior preserved all 32 parent prediction bytes while adding measured FLOPs. Full-32 adjusted gains remained negative, but expanded runtime coverage later found an outlier and prevented release.

Recorded measurement:

    Paidfirst128+SMoff+NPoff SCIfe8cadf3/RELd90e5299 now full32 verified both hosts: Mac-.187863082%all/-.164921093%held25, VM-.222178883%all/-.323998029%held25 versus actual337466. All32 own4da predictionbytes exact on each host. Paid extra13,629,046,784F total. RootVM independent95file/all12guards/pred/pairedratio review2f07dd3e in R21/ROOT_FIRST128_PAID_VM32_REVIEW_v1.json; ops6160af81. VMnativechild memoryevents/swap/OOM0. fe8 separated stage completed07:23:26Z, not oldfailedpartials.
    First128 base4da/e491 is fully verified on both hosts: Mac32 adjusted-.367205943%/held25-.347306137%; VM32-.401502959%/held-.506149506%, all paid bills included. RootMac custody9fb293c7; VM95file auditf5de32b7. Paid final first128+SMoff+NPoffSCIfe8cadf3/RELd90e5299 passed source/import/tiny and root quiet3 at06:46:20Z: all own4da bytes exact, actual residual time per35/150/200 -5.268080%/-2.770928%/-4.953876%, mean-4.330961%; instructionsmean-3.365156%. Actual extra bill+.182238% versus own4da. This is stronger measured runtime evidence, not hosted guarantee or a bank. Root started unchanged original Macparallel5 actual32 session34183 at06:58:00Z, output D/runtime_smoff_npxoff_first128_v1/science32_first128_smoff_npxoff_parallel5_v1; rootreview4b23f341. Ownedqueue77740 paused/drained and must resume finally; no other rootMac native.
    Paid first128 fe8/d90 expanded fixed13 completed08:26:21Z: meanactualresidual−0.858351691%,11/13lower, range−5.080467939% to+17.893281738%. Rank316 regressed41.015614ms despite instructions−1.104824%; preserve this outlier. Result0eb3a820; ROOT_FE8_EXPANDED13_RUNTIME_VERDICT_v1.json 8ffc63b6 holds source for insufficient consistent margin, no unchanged rerun. Existing full32 Mac−0.187863082% and VM−0.222178883% adjusted vs466 remain science evidence only. Own knowncold92–95 fe8 resultcbbe35d4 is−0.190763117% adjusted,raw−0.348019760%,bill+0.157709286%,2/4lower,SE0.587593%; custody88b54159 clean. Existing58c/d399 remains runtimeHOLD despite full32/knowncold sciencePASS.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/audit_native32_vm_custody_separated_v2.py](code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/audit_native32_vm_custody_separated_v2.py)
- [code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/build_v1.py](code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/build_v1.py)
- [code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/candidate_first128_smoff_npxoff_release_v1.py](code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/candidate_first128_smoff_npxoff_release_v1.py)
- [code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/audit_native32_vm_custody_v1.py](code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/audit_native32_vm_custody_v1.py)
- [code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/candidate_first128_smoff_npxoff_science_v1.py](code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/candidate_first128_smoff_npxoff_science_v1.py)
- [code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/check_import_v1.py](code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/check_import_v1.py)
- [code/astra_oct20_bill_v1/tail_npx_copy_exact_v1.py](code/astra_oct20_bill_v1/tail_npx_copy_exact_v1.py)
- [code/astra_oct20_bill_v1/tail_sm_direct_v1.py](code/astra_oct20_bill_v1/tail_sm_direct_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/tail_smoff_v1.py](code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/tail_smoff_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/tail_npxoff_v1.py](code/astra_oct20_vm_frontier_v1/runtime_npxoff_current337466_v1/tail_npxoff_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py)
- [code/fable_oct19_kc_v1/candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py](code/fable_oct19_kc_v1/candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6880](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6880)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6785](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6785)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7024](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7024)

<details>
<summary>All result files and records (27)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ROOT_FIRST128_PAID_VM32_REVIEW_v1.json](ROOT_FIRST128_PAID_VM32_REVIEW_v1.json) · result
- [ROOT_FE8_EXPANDED13_RUNTIME_VERDICT_v1.json](ROOT_FE8_EXPANDED13_RUNTIME_VERDICT_v1.json) · result
- [CURRENT_SOURCE7_v1.json](CURRENT_SOURCE7_v1.json) · result
- [VS_BEST_GRADED_v1.json](VS_BEST_GRADED_v1.json) · result
- [PARALLEL32_COMPLETE_v1.json](PARALLEL32_COMPLETE_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [RESULT32_v1.json](RESULT32_v1.json) · result
- [BUILD_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/BUILD_v1.json) · result
- [NATIVE_STAGE_COMPLETION_EVENTS_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/NATIVE_STAGE_COMPLETION_EVENTS_v1.json) · result
- [STRICT_WATCH_LAUNCH_separated_v2.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/STRICT_WATCH_LAUNCH_separated_v2.json) · result
- [STRICT_WATCH_INPUT_separated_v2.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/STRICT_WATCH_INPUT_separated_v2.json) · result
- [NATIVE_REVIEW_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/NATIVE_REVIEW_v1.json) · result
- [SCORING_PLAN_COMPATIBILITY_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/SCORING_PLAN_COMPATIBILITY_v1.json) · result
- [ROOT_SEPARATED_RECOVERY_REVIEW_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/ROOT_SEPARATED_RECOVERY_REVIEW_v1.json) · result
- [NATIVE_METADATA_COPY_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/NATIVE_METADATA_COPY_v1.json) · result
- [SCORING_PLAN_LINUX_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/SCORING_PLAN_LINUX_v1.json) · result
- [PREFLIGHT_FIRST_SQUARE_FIT_GUARD_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/PREFLIGHT_FIRST_SQUARE_FIT_GUARD_v1.json) · result
- [STATIC_RECOVERY_SCHEDULE_REVIEW_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/STATIC_RECOVERY_SCHEDULE_REVIEW_v1.json) · result
- [ROOT_SOURCE_REVIEW_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/ROOT_SOURCE_REVIEW_v1.json) · result
- [READY_SOURCE_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/READY_SOURCE_v1.json) · result
- [SEPARATED_BASELINE_CLARIFICATION_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/SEPARATED_BASELINE_CLARIFICATION_v1.json) · result
- [SHARED_MEMORY_FAILURE_REVIEW_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/SHARED_MEMORY_FAILURE_REVIEW_v1.json) · result
- [ROOT_MAC32_REVIEW_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/ROOT_MAC32_REVIEW_v1.json) · result
- [NATIVE_PLAN_LINUX_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/NATIVE_PLAN_LINUX_v1.json) · result
- [LOCAL_REPORT_CUSTODY_CONTINUATION_v1.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/LOCAL_REPORT_CUSTODY_CONTINUATION_v1.json) · result
- [SEPARATED_PREFLIGHT_v2.json](../../../records/research/2026-10-21-overlapping-24-capture-12-native-groups-exhaust-aggregate-memory/SEPARATED_PREFLIGHT_v2.json) · result

</details>

## Related work

- shipped in: [Submission #337506: First-128 readout with fully charged residual-time repair](../../../submissions/337506/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-paid-kernel-copy-restorations-preserve-first-128-predictions-reduce) · [Copy provenance](../../../PROVENANCE.json)

