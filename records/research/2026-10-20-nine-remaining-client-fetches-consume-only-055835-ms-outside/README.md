# Nine remaining client fetches consume only .055835 ms outside dispatch; blind deletion is invalid

2026-10-20T20:32:34Z · finding · informational · measured

Instrumented current-source execution found eight required roster checks and one required envelope check. Their measured outside-dispatch time did not explain the remaining residual budget pressure, and concurrent diagnostic timings were not qualification evidence.

Recorded measurement:

    RESIDUAL THREAD DIAGNOSTIC: D/boundary_sync_run_v2.py exactcurrentdualenv, D/sync_diag_micro_4a40_r35_v2 completedunderaux30. Nineactualfetches (8DCRroster+1SLRenvelope), zero float/int/item/tolist;boolcalls2.348542ms total,2.292707msinlegitimatedispatch,only.055835msoutside. Nohiddenlargescalarflushstall/no scorecredit. DCRprefixcheckrequiredforties; SLRrequired; no blindremoval. D/SYNC_CURRENT_AUDIT_v1.json. DiagnosticTIMINGS/PREDICTIONSNOTqualification: concurrentinstrumentation activatedinheritedfastrescue atL12/32calls. Quietqualificationbothsources had0rescues.

## Programs

- [boundary_sync_run_v2.py](boundary_sync_run_v2.py)
- [../../../experiments/arithmetic-kernels/retained-cumulant-gate-centered-channel-core-reuse-padding-0/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/thread_counts_v3.py](../../../experiments/arithmetic-kernels/retained-cumulant-gate-centered-channel-core-reuse-padding-0/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/dual_thread_boundary_v3/thread_counts_v3.py)
- [boundary_sync_client_v2.py](boundary_sync_client_v2.py)
- [sync_observer_v1.py](sync_observer_v1.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py](../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6280](../../TEAM_LOG_2026-10-01_to_10-22.md#L6280)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SYNC_CURRENT_AUDIT_v1.json](SYNC_CURRENT_AUDIT_v1.json) · result
- [launcher.json](launcher.json) · result
- [server.json](server.json) · result
- [end_gc_stress.json](end_gc_stress.json) · result
- [manifest.json](manifest.json) · result
- [sync_diagnostic.json](sync_diagnostic.json) · result
- [GUARD_v1.json](GUARD_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-nine-remaining-client-fetches-consume-only-055835-ms-outside) · [Copy provenance](../../../PROVENANCE.json)

