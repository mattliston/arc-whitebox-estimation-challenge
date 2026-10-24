# Setup prebind guard makes local FLOP counts load-dependent; release uses deterministic count sessions

2026-10-12T18:33Z · finding · informational · derived

The same file re-run under load reads a different in-process metered FLOPs because the setup's wall-clock prebind guard binds fewer keys (lazily bound later and billed), so release's fast gate takes metered FLOPs and requests from the deterministic platform-pace count sessions and judges only the physics natively; the box itself was power-capped to 6-10 of 16 cores.

Recorded measurement:

    rehearsal twin: predictions bit-identical 10/10 but native bill +0.199% where the fire run read -0.155%; network 651 +814 ops / +0.56% flops; kern.sched_recommended_cores mostly 8

## Programs

- [../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/fastgate_v2.sh](../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/fastgate_v2.sh)
- [bill_watch_v1.py](bill_watch_v1.py)
- [../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/compare_twin_byteid_v1.py](../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/compare_twin_byteid_v1.py)
- [../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/quick.py](../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/quick.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/fastgate_pin_v1.py](../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/fastgate_pin_v1.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/compare_twin_native_v4.py](../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/compare_twin_native_v4.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py)
- [../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py](../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1582](../../TEAM_LOG_2026-10-01_to_10-22.md#L1582)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1618](../../TEAM_LOG_2026-10-01_to_10-22.md#L1618)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1627](../../TEAM_LOG_2026-10-01_to_10-22.md#L1627)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1726](../../TEAM_LOG_2026-10-01_to_10-22.md#L1726)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1737](../../TEAM_LOG_2026-10-01_to_10-22.md#L1737)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Combined-candidate workflow applies verified changes, seals on fresh networks and runs a fast release gate](../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/README.md)
- enables: [Submission #335621: higher-request form of the same refit, qualified through the fast release gate](../../../submissions/335621/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-setup-prebind-guard-makes-local-flop-counts-load-dependent) · [Copy provenance](../../../PROVENANCE.json)

