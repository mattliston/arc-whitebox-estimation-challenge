# Request caches and seven-level strip pushes adopted: byte-identical request savings and about -0.18% FLOPs

2026-10-11T06:02Z · decision · adopted · measured

A request census found repeated views and conversions worth about 500 requests per network at identical metered FLOPs; seven-level Winograd strip pushes on full 2048-column packets pay about -0.18% of metered FLOPs at rounding level, while seven-level reads do not pay (their transforms eat the leaf saving).

Recorded measurement:

    XRQ -459 / -490 requests, bill identical, predictions byte-identical; XRQ2 charged ops -191 / -203, bill -0.011%; HZ7R bill -0.249 / -0.146% (mean -0.198%), ops +162 / +38; 7-level reads +0.01..+13% per product

## Programs

- [run_sites.sh](run_sites.sh)
- [sites_table.py](sites_table.py)
- [probe_repeat.py](probe_repeat.py)
- [../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a56_req/fragment_rqd_v1.py](../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a56_req/fragment_rqd_v1.py)
- [../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a57_hz7/fragment_hz7_v1.py](../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a57_hz7/fragment_hz7_v1.py)
- [../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a57_hz7/fragment_hz7_v4.py](../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a57_hz7/fragment_hz7_v4.py)
- [fragment_hz7_v2.py](fragment_hz7_v2.py)
- [unit_hz7.py](unit_hz7.py)
- [../../../instruments/native40-scorer-seals-85-calls-requires-lower-own-bill/code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py](../../../instruments/native40-scorer-seals-85-calls-requires-lower-own-bill/code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../experiments/flop-accounting/non-product-flop-census-exact-request-neutral-changes-offer/code/census/census2.py](../../../experiments/flop-accounting/non-product-flop-census-exact-request-neutral-changes-offer/code/census/census2.py)

<details>
<summary>Additional programs and dependencies (8)</summary>

- [../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/quick.py](../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/quick.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py)
- [../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py](../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1004](../../TEAM_LOG_2026-10-01_to_10-22.md#L1004)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1006](../../TEAM_LOG_2026-10-01_to_10-22.md#L1006)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1008](../../TEAM_LOG_2026-10-01_to_10-22.md#L1008)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1011](../../TEAM_LOG_2026-10-01_to_10-22.md#L1011)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1014](../../TEAM_LOG_2026-10-01_to_10-22.md#L1014)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1016](../../TEAM_LOG_2026-10-01_to_10-22.md#L1016)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [FLOP census: exact kernel changes each save at most 0.2%; live-floor push cut adds 13% error](../../../experiments/flop-accounting/flop-census-exact-kernel-changes-each-save-most-0/README.md)
- enables: [Persistent sandwich scratch, direct strip loads and true-inner-size products adopted: -0.200/-0.284% FLOPs](../2026-10-11-persistent-sandwich-scratch-direct-strip-loads-true-inner-size/README.md)
- shipped in: [Submission #335329 fails: network 14 exceeds the residual cap at 0.4083 s](../../../submissions/335329/README.md)
- shipped in: [Submission #335340: ridge-re-sized refit, request caches and seven-level full-packet strip pushes](../../../submissions/335340/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-request-caches-seven-level-strip-pushes-adopted-byte-identical) · [Copy provenance](../../../PROVENANCE.json)

