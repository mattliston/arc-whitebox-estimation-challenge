# Join-sketch oversampling 64/192: first-seven +1.627310% versus #337393; full-32 test reprioritized

2026-10-20T20:59:42Z · experiment · rejected · measured

The previously untested wider intermediate sketch failed the economic screen on seven networks. It was retired after completing active groups, so the record does not treat it as a completed 32-network rejection.

Recorded measurement:

    NT64/q192 first7FAILS economically: +1.627310%vs337393,SE.694816,1/7lower,raw+.364935%,bill+1.258164%. Q192 improvesq128first7-.273340% butstillnotcompetitive. EvenhypotheticalremovalofENTIRENTincrementleaves+.638%vsbest. D/NT64_Q192_FIRST7_COST_CLOSURE_v1.json. HeldCOPY_ALLcompaniona6a892dd/9f53c72f remainsUNQUEUED. ExactNTdispatcher2343 stopped20:58:16Z; letcurrentguardgroupfinishthenretireonlydispatcher, preserveallparts. K/retire_nt_for_gate_coadapt_v1.py session68788; oncompletionABORTED_PRIORITY_v1.json. This isfirst7reprioritization, NOTcompleted32rejection.
    NT64/q192 gap found: old current32 NT32/64 closures used q128. Exact screened p64/q192 source was never current32 judged. New ntc64_q192_nogov_astra SCIc25a89a1e05615730563bd472d2e70b6445781c56d3eff5e485541ea53033f4a REL0ef3603c14dc578858cf4a92df2721ccf74d8d7969ef53ba495579fbd94fe3d6 adds only original tail_ntc64_all and permanent disable. Historical fresh7100015-18 versus OLDv4a mean-1.197605% SE.888974,3/4lower,raw-2.164728%,bill+.988166%; gain versus NEWBEST unknown. D/NT64_Q192_CURRENT_GAP_v1.json. Queueprepend took effect at NEXT free primary slot; actual launch PID2343 ~20:46Z. CORRECTION: PID95263 is micro_k31age4_ecut_astra, NOT NT. Agent preparing held exactCOPY_ALL companion; no additive gain assumed.
    NTq192 dispatcher2343 cleanly retired20:59:48Z after all current workers completed;5 groups preserved, ABORTED_PRIORITY receipt. Primary queue77740 capacity5 with one gate_coadapt reservation. Fresh coadapt native2 partitions launched21:07:34Z(primary session60920) and21:07:36Z(aux session48504), seeds92/93 and94/95. Fourprimary32 judges+oneprimarynative+oneguard30auxnative; existing guards/admissions untouched. Actual paired337393 bank then6 cached-live GSK/GK4 settings on microedge; bank/source/guard/truthpins verified. Logs A/gate_coadapt_v1. No new gain assumed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/retire_nt_for_gate_coadapt_v1.py](code/retire_nt_for_gate_coadapt_v1.py)
- [code/agent_astra_joint_v1/gate_coadapt_v1/analyze_v2.py](code/agent_astra_joint_v1/gate_coadapt_v1/analyze_v2.py)
- [code/agent_astra_joint_v1/gate_coadapt_v1/run_partition_v2.py](code/agent_astra_joint_v1/gate_coadapt_v1/run_partition_v2.py)
- [code/agent_astra_joint_v1/gate_coadapt_v1/check_cache_ast_v1.py](code/agent_astra_joint_v1/gate_coadapt_v1/check_cache_ast_v1.py)
- [code/agent_astra_joint_v1/gate_coadapt_v1/diagnose_response_v1.py](code/agent_astra_joint_v1/gate_coadapt_v1/diagnose_response_v1.py)
- [code/agent_astra_joint_v1/gate_coadapt_v1/run_partition_v1.py](code/agent_astra_joint_v1/gate_coadapt_v1/run_partition_v1.py)
- [code/agent_astra_joint_v1/gate_coadapt_v1/analyze_v1.py](code/agent_astra_joint_v1/gate_coadapt_v1/analyze_v1.py)
- [code/candidate_micro_copyall_ecut_nogov_astra_sci_v1.py](code/candidate_micro_copyall_ecut_nogov_astra_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6304](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6304)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6291](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6291)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6311](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6311)

<details>
<summary>All result files and records (13)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [NT64_Q192_FIRST7_COST_CLOSURE_v1.json](NT64_Q192_FIRST7_COST_CLOSURE_v1.json) · result
- [NT64_Q192_CURRENT_GAP_v1.json](NT64_Q192_CURRENT_GAP_v1.json) · result
- [README_v1.md](README_v1.md) · record
- [settings_v1.json](settings_v1.json) · result
- [MANIFEST_v1.json](MANIFEST_v1.json) · result
- [MANIFEST_v2.json](MANIFEST_v2.json) · result
- [RESPONSE_DIAGNOSTIC_v1.json](RESPONSE_DIAGNOSTIC_v1.json) · result
- [CACHE_AST_PROOF_v1.json](CACHE_AST_PROOF_v1.json) · result
- [bank_settings_v1.json](bank_settings_v1.json) · result
- [V2_LOG_REPAIR_PROOF_v1.json](V2_LOG_REPAIR_PROOF_v1.json) · result
- [FROZEN_SELECTION_GK4114_v1.json](FROZEN_SELECTION_GK4114_v1.json) · result
- [RESULT_FRESH4_v1.json](RESULT_FRESH4_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-join-sketch-oversampling-64-192-first-seven-1-627310) · [Copy provenance](../../../PROVENANCE.json)

