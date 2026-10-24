# Residual-loss model recalibrated to 7.35 microseconds per request after a 1.4-1.7% median shortfall

2026-10-11T20:57:47Z · finding · informational · measured

Release checked loss_model_v3 (per-network residual factors from our graded runs) for network-size coverage: over the 10 graded candidate generation 44 (S44)-candidate generation 48 (S48) runs the per-network structural factor spans 0.973-1.073 (heaviest network 7), the metered FLOPs factor 0.965-1.073, residual follows metered FLOPs only weakly (correlation 0.28, +0.13% residual per +1% metered FLOPs) and the transient tail over a network's own level has p99 1.116 and max 1.306. The model's median (0.99 x network 100's count x 7.31 us) ran 1.4-1.7% low against the graded medians of #335340 / #335349 / #335411 (7.34-7.36 us per request), so every P(loss) row was re-priced at 7.35 us. The warm cap judged on network 100 alone understates the heaviest warm network 39 (42,695 requests on S48b, 5 under the 42,700 cap), so the gate script added an informational network-39 session.

Recorded measurement:

    structural factor 0.973-1.073 (p10 0.984, p90 1.013), bill factor 0.965-1.073, correlation 0.28, transient tail p99 1.116, p99.9 1.233, max 1.306; model median 1.4-1.7% low (graded 7.34-7.36 us vs 7.31 us); re-priced S48 safe 3.0% per 100 / 8.8% per 300, S49 safe 2.7% / 8.0%, S49 twin (1)+(3) 4.2% / 12.0%; network 39 on S48b 42,695 requests (bill 269.49 G, +7% over 35 / 100 / 37); S48 twin on 39: 44,277 (over the warm cap by 1577)

## Programs

- [../2026-10-11-residual-time-model-has-heavy-tail-retain-free-request/loss_model_v3.py](../2026-10-11-residual-time-model-has-heavy-tail-retain-free-request/loss_model_v3.py)
- [s47_safe_v7.sh](s47_safe_v7.sh)
- [../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/single-attempt-submission-uploader-records-one-qualified-archive-receipt/code/fire_once.py](../../../instruments/single-attempt-submission-uploader-records-one-qualified-archive-receipt/code/fire_once.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py)
- [../../../instruments/pair-qualification-script-checks-both-implementations-through-native-slow/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/build_variant.py](../../../instruments/pair-qualification-script-checks-both-implementations-through-native-slow/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/build_variant.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py](../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)

<details>
<summary>Additional programs and dependencies (12)</summary>

- [../2026-10-01-concurrent-workers-exhaust-memory-compression-capacity-require-reduced/own_gate.sh](../2026-10-01-concurrent-workers-exhaust-memory-compression-capacity-require-reduced/own_gate.sh)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py](../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py)
- [../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py](../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py)
- [../2026-10-11-first-network-request-cap-replaced-modelled-loss-risk-10/release_gate_v4.py](../2026-10-11-first-network-request-cap-replaced-modelled-loss-risk-10/release_gate_v4.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py)
- [../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py](../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py)
- [../../../instruments/pair-qualification-script-checks-both-implementations-through-native-slow/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_a_package.py](../../../instruments/pair-qualification-script-checks-both-implementations-through-native-slow/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_a_package.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/stage_c_boundary.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/stage_c_boundary.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)
- [candidate.py](candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1203](../../TEAM_LOG_2026-10-01_to_10-22.md#L1203)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1204](../../TEAM_LOG_2026-10-01_to_10-22.md#L1204)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1205](../../TEAM_LOG_2026-10-01_to_10-22.md#L1205)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1239](../../TEAM_LOG_2026-10-01_to_10-22.md#L1239)

<details>
<summary>All result files and records (11)</summary>

- [busy_at_start.json](busy_at_start.json) · result
- [guard.json](guard.json) · result
- [guard.log](guard.log) · log
- [launcher.json](launcher.json) · result
- [manifest.json](manifest.json) · result
- [rows.jsonl](rows.jsonl) · result
- [server.json](server.json) · result
- [server.log](server.log) · log
- [server_marks.jsonl](server_marks.jsonl) · result
- [stage_c.json](stage_c.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-residual-loss-model-recalibrated-7-35-microseconds-per-request) · [Copy provenance](../../../PROVENANCE.json)

