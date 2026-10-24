# First runtime profiler perturbs free batching and loses its output artifact despite exact predictions

2026-10-21T07:47:44Z · incident · informational · measured

The observer preserved predictions, FLOPs and operations but changed free-trip counts from 4627 to 4271. A scope error also lost the profile artifact; the run was rejected and all evidence retained.

Recorded measurement:

    Exactd399 profile211556 retired after both originalguards completed clean (control21156107:41:29Z,observer21182007:43:17Z), zero childmemoryevents/Swap/rescue, peaks4.747/4.785GB. `V/profile_first128_d399_v1/PROFILE_FAILURE_AUDIT_v1.json` SHAc135e2447b17f69c6297fbd0d622f88ac4420b96c8d9c9af9c0f10082a88239d rehashes19custodyfiles. Predictions65536B matchfreshcontrol andown4da exactly,e97f6fe0; both235862163509F,43581requests,42230freedhandles,identicaloperationinvoice. Requiredfree-trip equality FAILED4627→4271, so diagnostic REJECTED and no COMPLETE/runtime/sourcehotspot claim. A separate adapter defect lost profileartifact: observerlocaltoevaluate(), butmodulefinallylooks inglobals(). No retry launched. Any revisedobserver mustfixscope, investigateallocation-sensitivefreebatching, preserveallstrictgatesandreceive root/slotreview. Controllerandbothgroupsretired;opsnotifiedslotavailable.

## Programs

- [run_group_profile_v1.py](run_group_profile_v1.py)
- [client_profile_v1.py](client_profile_v1.py)
- [run_diagnostic_v1.py](run_diagnostic_v1.py)
- [check_profile_metadata_v1.py](check_profile_metadata_v1.py)
- [../../../instruments/linux-native-worker-port-verifies-official-source-paired-controls/code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](../../../instruments/linux-native-worker-port-verifies-official-source-paired-controls/code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [../../../instruments/linux-native-worker-port-verifies-official-source-paired-controls/code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](../../../instruments/linux-native-worker-port-verifies-official-source-paired-controls/code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [profile_outside_dispatch_v1.py](profile_outside_dispatch_v1.py)
- [_dispatch.py](_dispatch.py)
- [../../../instruments/linux-native-worker-port-verifies-official-source-paired-controls/code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](../../../instruments/linux-native-worker-port-verifies-official-source-paired-controls/code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)
- [../../../instruments/linux-native-worker-port-verifies-official-source-paired-controls/code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/server_linux_v7.py](../../../instruments/linux-native-worker-port-verifies-official-source-paired-controls/code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/server_linux_v7.py)
- [client_linux_markline_v1.py](client_linux_markline_v1.py)
- [../../../experiments/birth-terms/live-birth-consumer-coefficient-refit-passes-teacher-fidelity-but/code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py](../../../experiments/birth-terms/live-birth-consumer-coefficient-refit-passes-teacher-fidelity-but/code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [run_group_linux_markline_v1.py](run_group_linux_markline_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6913](../../TEAM_LOG_2026-10-01_to_10-22.md#L6913)

<details>
<summary>All result files and records (11)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [PROFILE_FAILURE_AUDIT_v1.json](PROFILE_FAILURE_AUDIT_v1.json) · result
- [METADATA_CHECK_py310_v1.json](METADATA_CHECK_py310_v1.json) · result
- [CLOSED_LEDGER_SOURCE_REVIEW_v1.json](CLOSED_LEDGER_SOURCE_REVIEW_v1.json) · result
- [PLAN_LINUX_v1.json](PLAN_LINUX_v1.json) · result
- [ROOT_DIAGNOSTIC_SLOT_REVIEW_v1.json](ROOT_DIAGNOSTIC_SLOT_REVIEW_v1.json) · result
- [CLOSED_LEDGER_SOURCE_REVIEW_v2.json](CLOSED_LEDGER_SOURCE_REVIEW_v2.json) · result
- [LAUNCH_v1.json](LAUNCH_v1.json) · result
- [ROOT_STATIC_REVIEW_v1.json](ROOT_STATIC_REVIEW_v1.json) · result
- [RECIPE_v1.json](RECIPE_v1.json) · result
- [STATIC_REVIEW_v1.json](STATIC_REVIEW_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-first-runtime-profiler-perturbs-free-batching-loses-output-artifact) · [Copy provenance](../../../PROVENANCE.json)

