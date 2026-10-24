# Robust roster scatter and backend overflow predicate remove materializations but fail runtime margin

2026-10-21T08:07:37Z · experiment · rejected · measured

The robust scatter preserved finite-precision roster ties and ownership, while a paid backend predicate preserved overflow errors. Full-32 accuracy remained exact to the parent, but 13-network runtime evidence had only -2.562521% mean improvement and one regression.

Recorded measurement:

    Final paid first128 SCI58c02395/RELd3991b2f full32 passes both hosts: Mac actualadjusted vs466 -.164032509%all/-.141290869%held25,22/32lower, independent audit abbb5f03. Mac maximum system swap growth596702658.56bytes within original conditional guard; not zero. VM actual -.198358779%all/-.300408378%held25,20/32lower, root95file/all12guards de2b1c5d and ops fb831ebd. All own4da/fe8 prediction bytes exact. Runtime add-on paid total15,437,072,107F above4da; includes1,808,025,323F abovefe8. Root quiet fixed13 completed07:37:07Z, receipt7f33e9d0: meanactualresidual -2.562521%,12/13lower, range-4.588960% to+0.339338%; rank200regressed. ROOT RUNTIME HOLD: insufficient consistent margin, no unchanged timing retry. Mac32completed07:46:37Z, original77740queue restored. No root Mac native active asof08:05.
    Exact final stack `D/runtime_first128_dcr_slr_v1` SCI58c02395cd6419ae79bbd371c912ca44bd6d45f7552d93a63c087da5a5d141f2 / RELd3991b2ffd6bf3f5ed4375257972d813cac4da45a27682bc11649f368c6ce0b7. DCR parent prefix unchanged; only SLR final guard expression changed. Full2 importsPASS guard203521; source READYda84a324, root independent proof885a2309. DCR-only packet remainsimmutablefallback. Rootquiet session6286 started07:12:26Z and completed all3exactown4da bytes; actual meanresidual−3.378539%,instructions−3.083395%, paidbill+0.204608% vsphysics; rank200 only−0.775%. Earlier fe8mean−4.331% uses differentfreshcontrol, so cross-runmeans do NOT prove DCRcausedloss. Current root decision open, NOFIRE. Native planb57cc419/review5206367a is original32 canonicalmarkline with bank337393 metadata; ops will launch only reviewed separatedstaticstage afterfe8.
    Combined candidate SCI58c02395/RELd3991b2f adds proven DCR robust scatter and failure-preserving SLR Cholesky predicate guard. Rootsource885a2309, READYda84a324. Quiet3 completed07:16:56Z: mean actual residual-3.378539%, instructions-3.083395%; rank200 only-.775122% residual, so coverage expanded before any release. Root fixed13 original first5groups running session95434 from07:20:09Z; queue77740 paused/drained and restoredfinally. ROOT_REVIEW67a6a5e0, exactplan d6916322, all original guards. Do not launch another Macnative. Partial13 is not a verdict. Final58c actualMac32 not launched; prepared mac32_packet_v1/READY_MAC32f6f8874a, unchanged wrapperffce765a and audit1f3e1bd6. Needs fullcoveragejudgment first.
    `CLOSED_LEDGER_SOURCE_REVIEW_v1.json` preserves read-only exactsource/closedidea audit. Generic GC/wrapper/stats/compact/DAG/prefix families remain closed absent a specific newly measured hotspot. DCR cheap-prefix provenance is unproved: live selection stably sorts roundedCDF(a), deadcut sorts rounded(a+Z); monotonicity cannot prove identical finite-precision tie ordering. Keep robust scatter. Root full13coverage result7f33e9d0 meanactualresidual−2.562521%,range−4.58896 to+0.339338%,12/13lower; root holds release/fire. Root final58c Mac32 session94032 began07:38:52Z for actual paid accuracy, not runtime proof; this lane has not launched Mac.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/tail_slr_cholesky_v1.py](code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/tail_slr_cholesky_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/candidate_first128_smoff_npxoff_dcr_slr_science_v1.py](code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/candidate_first128_smoff_npxoff_dcr_slr_science_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/candidate_first128_smoff_npxoff_dcr_slr_release_v1.py](code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/candidate_first128_smoff_npxoff_dcr_slr_release_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/build_v1.py](code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/build_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/audit_native32_vm_custody_v1.py](code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/audit_native32_vm_custody_v1.py)
- [code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/check_import_v1.py](code/astra_oct20_bill_v1/runtime_first128_dcr_slr_v1/check_import_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6955](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6955)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6855](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6855)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6882](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6882)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6908](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6908)

<details>
<summary>All result files and records (16)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [BUILD_v1.json](BUILD_v1.json) · result
- [STRICT_WATCH_LAUNCH_v1.json](STRICT_WATCH_LAUNCH_v1.json) · result
- [CHECKS_LAUNCH_v1.json](CHECKS_LAUNCH_v1.json) · result
- [NATIVE_STAGE_COMPLETION_EVENTS_v1.json](NATIVE_STAGE_COMPLETION_EVENTS_v1.json) · result
- [NATIVE_STAGE_METADATA_DEPLOYED_v1.json](NATIVE_STAGE_METADATA_DEPLOYED_v1.json) · result
- [IMPORT_CHECK_v1.json](IMPORT_CHECK_v1.json) · result
- [STATIC_NATIVE_STAGE_REVIEW_v1.json](STATIC_NATIVE_STAGE_REVIEW_v1.json) · result
- [STRICT_WATCH_INPUT_v1.json](STRICT_WATCH_INPUT_v1.json) · result
- [NATIVE_STAGE_PREFLIGHT_v1.json](NATIVE_STAGE_PREFLIGHT_v1.json) · result
- [NATIVE_REVIEW_v1.json](NATIVE_REVIEW_v1.json) · result
- [ROOT_SOURCE_REVIEW_v1.json](ROOT_SOURCE_REVIEW_v1.json) · result
- [READY_SOURCE_v1.json](READY_SOURCE_v1.json) · result
- [ROOT_NATIVE_STAGE_REVIEW_v1.json](ROOT_NATIVE_STAGE_REVIEW_v1.json) · result
- [NATIVE_PLAN_LINUX_v1.json](NATIVE_PLAN_LINUX_v1.json) · result
- [GUARD_v1.json](GUARD_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-robust-roster-scatter-backend-overflow-predicate-remove-materializations-but) · [Copy provenance](../../../PROVENANCE.json)

