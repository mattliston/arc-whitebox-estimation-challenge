# Paired native measurement: higher-request variant adds 2.34% requests without reducing FLOPs

2026-10-11T04:44Z · finding · informational · measured

Both submitted candidate generation 46 (S46) files were re-measured in one paired boundary session (legs P C C P, setup under the platform clock with the constants charged): the safe #335259 at 43,398 / 42,543 requests (first / warm), 45,639 residual instructions per request and 254.65 Gflop net metered FLOPs over 100 networks; the twin #335261 at 44,537 / 43,417 requests, 46,423 instructions per request and 254.70 Gflop; the twin's projected residual from the safe file's graded page is 0.313 s mean / 0.351 s worst at 77.5 / 85.8 s wall. Counts and instructions do not depend on machine load, only the local timings do.

Recorded measurement:

    safe 43,398 / 42,543 requests, 45,639 instr/request, 254.65 Gflop; twin 44,537 / 43,417, 46,423 instr/request, 254.70 Gflop; ratios requests 1.0234, bill 1.00024; twin projected residual 0.313 / 0.351 s, wall 77.5 / 85.8 s

## Programs

- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py)
- [../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py](../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py)
- [../../../submissions/335259/candidate.py](../../../submissions/335259/candidate.py)
- [../../../submissions/335261/candidate.py](../../../submissions/335261/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1005](../../TEAM_LOG_2026-10-01_to_10-22.md#L1005)

<details>
<summary>All result files and records (38)</summary>

- [busy_at_start.json](busy_at_start.json) · result
- [guard.json](guard.json) · result
- [guard.log](guard.log) · log
- [launcher.json](launcher.json) · result
- [manifest.json](manifest.json) · result
- [rows.jsonl](rows.jsonl) · result
- [server.json](server.json) · result
- [server.log](server.log) · log
- [server_marks.jsonl](server_marks.jsonl) · result
- [busy_at_start-926bba23.json](busy_at_start-926bba23.json) · result
- [guard-1ca0c83e.json](guard-1ca0c83e.json) · result
- [guard-993ade64.log](guard-993ade64.log) · log
- [launcher-546910c6.json](launcher-546910c6.json) · result
- [manifest-7bfa833c.json](manifest-7bfa833c.json) · result
- [rows-159338de.jsonl](rows-159338de.jsonl) · result
- [server-cf477a77.json](server-cf477a77.json) · result
- [server-d00801f8.log](server-d00801f8.log) · log
- [server_marks-f44c4db0.jsonl](server_marks-f44c4db0.jsonl) · result
- [busy_at_start-3dc65b57.json](busy_at_start-3dc65b57.json) · result
- [guard-f7ceadb7.json](guard-f7ceadb7.json) · result
- [guard-16a58538.log](guard-16a58538.log) · log
- [launcher-98060ead.json](launcher-98060ead.json) · result
- [manifest-3c97041f.json](manifest-3c97041f.json) · result
- [rows-497b4cc1.jsonl](rows-497b4cc1.jsonl) · result
- [server-067437fc.json](server-067437fc.json) · result
- [server-4e6482b2.log](server-4e6482b2.log) · log
- [server_marks-c41ccdb7.jsonl](server_marks-c41ccdb7.jsonl) · result
- [busy_at_start-da073b9a.json](busy_at_start-da073b9a.json) · result
- [guard-7e7b4cd3.json](guard-7e7b4cd3.json) · result
- [guard-0aaa96d3.log](guard-0aaa96d3.log) · log
- [launcher-4949255d.json](launcher-4949255d.json) · result
- [manifest-7ded1179.json](manifest-7ded1179.json) · result
- [rows-8df25082.jsonl](rows-8df25082.jsonl) · result
- [server-ca21f359.json](server-ca21f359.json) · result
- [server-7dfa231f.log](server-7dfa231f.log) · log
- [server_marks-787b81f5.jsonl](server_marks-787b81f5.jsonl) · result
- [stage_c.json](stage_c.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-paired-native-measurement-higher-request-variant-adds-2-34) · [Copy provenance](../../../PROVENANCE.json)

