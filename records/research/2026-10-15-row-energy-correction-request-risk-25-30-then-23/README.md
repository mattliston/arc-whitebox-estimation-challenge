# Row-energy correction request risk 25.30% then 23.41%; split-view batching reaches 19.60%

2026-10-15T22:46:41Z · incident · adopted · measured

The original row-energy release and first exact view repair both exceeded the request-risk limit. Splitting and batching views preserved ten-network predictions/FLOPs and passed the original limit.

Recorded measurement:

    risk25.30% first48826/warm46711;firstrepair23.41% warm46578;final19.60%;native10all16bytes/FLOPsexact

## Programs

- [export_prefix22_science_ten_v1.py](export_prefix22_science_ten_v1.py)
- [finalize_metadata_v1.py](finalize_metadata_v1.py)
- [finalize_metadata_v1-0836961b.py](finalize_metadata_v1-0836961b.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_prefix22_uvp_split_HELD_v1.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_prefix22_uvp_split_HELD_v1.py)
- [check_tiny_v1.py](check_tiny_v1.py)
- [run_remote_tiny_v1.py](run_remote_tiny_v1.py)
- [finalize_metadata_v1-f15748b0.py](finalize_metadata_v1-f15748b0.py)
- [finish_split_release_v1.py](finish_split_release_v1.py)
- [../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/single-attempt-submission-uploader-records-one-qualified-archive-receipt/code/fire_once.py](../../../instruments/single-attempt-submission-uploader-records-one-qualified-archive-receipt/code/fire_once.py)

<details>
<summary>Additional programs and dependencies (13)</summary>

- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py](../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py)
- [../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/fastgate_pin_v1.py](../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/fastgate_pin_v1.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py](../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py)
- [../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py](../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py)
- [../2026-10-11-first-network-request-cap-replaced-modelled-loss-risk-10/release_gate_v4.py](../2026-10-11-first-network-request-cap-replaced-modelled-loss-risk-10/release_gate_v4.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py)
- [../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py](../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3557](../../TEAM_LOG_2026-10-01_to_10-22.md#L3557)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3553](../../TEAM_LOG_2026-10-01_to_10-22.md#L3553)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3555](../../TEAM_LOG_2026-10-01_to_10-22.md#L3555)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3559](../../TEAM_LOG_2026-10-01_to_10-22.md#L3559)

<details>
<summary>All result files and records (38)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ACTUAL_MANAGED_EXIT_normal_v1.json](ACTUAL_MANAGED_EXIT_normal_v1.json) · result
- [ACTUAL_MANAGED_EXIT_slow_v1.json](ACTUAL_MANAGED_EXIT_slow_v1.json) · result
- [DELIVERY_READY_v1.json](DELIVERY_READY_v1.json) · result
- [FINALIZATION_SEQUENCE_HELD_v1.json](FINALIZATION_SEQUENCE_HELD_v1.json) · result
- [FRESH_ALGEBRA_PEER_RELAY_v1.json](FRESH_ALGEBRA_PEER_RELAY_v1.json) · result
- [MANAGED_SESSION_normal_v1.json](MANAGED_SESSION_normal_v1.json) · result
- [MANAGED_SESSION_slow_v1.json](MANAGED_SESSION_slow_v1.json) · result
- [NATIVE_TEN_SCIENCE_EXPORT_v1.json](NATIVE_TEN_SCIENCE_EXPORT_v1.json) · result
- [OWNED_PROCESS_SNAPSHOT_v1.json](OWNED_PROCESS_SNAPSHOT_v1.json) · result
- [RELEASE_STOPPED_v1.json](RELEASE_STOPPED_v1.json) · result
- [ACTUAL_MANAGED_EXIT_normal_v1-1bbf9b65.json](ACTUAL_MANAGED_EXIT_normal_v1-1bbf9b65.json) · result
- [ACTUAL_MANAGED_EXIT_slow_v1-61c10ddd.json](ACTUAL_MANAGED_EXIT_slow_v1-61c10ddd.json) · result
- [DELIVERY_READY_v1-f7131f51.json](DELIVERY_READY_v1-f7131f51.json) · result
- [FINALIZATION_SEQUENCE_HELD_v1-22eecd2e.json](FINALIZATION_SEQUENCE_HELD_v1-22eecd2e.json) · result
- [MANAGED_SESSION_normal_v1-814ce555.json](MANAGED_SESSION_normal_v1-814ce555.json) · result
- [MANAGED_SESSION_slow_v1-36d60445.json](MANAGED_SESSION_slow_v1-36d60445.json) · result
- [OWNED_PROCESS_RETIREMENT_SNAPSHOT_v1.json](OWNED_PROCESS_RETIREMENT_SNAPSHOT_v1.json) · result
- [RELEASE_STOPPED_v1-30da05a9.json](RELEASE_STOPPED_v1-30da05a9.json) · result
- [ROOT_VECTOR_EINSUM_REVIEW_v1.json](ROOT_VECTOR_EINSUM_REVIEW_v1.json) · result
- [ROOT_VIEW_REPAIR_AUTHORITY_v1.json](ROOT_VIEW_REPAIR_AUTHORITY_v1.json) · result
- [STAGE_A_EXACT_ARCHIVE_v1.json](STAGE_A_EXACT_ARCHIVE_v1.json) · result
- [FRESH_ALGEBRA_SOURCE_PEER_v1.json](FRESH_ALGEBRA_SOURCE_PEER_v1.json) · result
- [SOURCE_HELD_v1.json](SOURCE_HELD_v1.json) · result
- [ACTUAL_FNP_TINY_v1.json](ACTUAL_FNP_TINY_v1.json) · result
- [REMOTE_TINY_v1.json](REMOTE_TINY_v1.json) · result
- [SPLIT_OWNED_EXIT_v1.json](SPLIT_OWNED_EXIT_v1.json) · result
- [SPLIT_SERVER_v1.json](SPLIT_SERVER_v1.json) · result
- [RELEASE_CONFIRMED_v1.json](RELEASE_CONFIRMED_v1.json) · result
- [ACTUAL_MANAGED_EXIT_normal_v1-90c1ed1a.json](ACTUAL_MANAGED_EXIT_normal_v1-90c1ed1a.json) · result
- [ACTUAL_MANAGED_EXIT_slow_v1-059ba6a7.json](ACTUAL_MANAGED_EXIT_slow_v1-059ba6a7.json) · result
- [DELIVERY_READY_v1-2a0b0a9f.json](DELIVERY_READY_v1-2a0b0a9f.json) · result
- [DELIVERY_READY_v2.json](DELIVERY_READY_v2.json) · result
- [FINALIZATION_COPY_01_v1.json](FINALIZATION_COPY_01_v1.json) · result
- [FINALIZATION_COPY_03_v1.json](FINALIZATION_COPY_03_v1.json) · result
- [FINALIZATION_SEQUENCE_HELD_v1-da1c052e.json](FINALIZATION_SEQUENCE_HELD_v1-da1c052e.json) · result
- [MANAGED_SESSION_normal_v1-0fd42712.json](MANAGED_SESSION_normal_v1-0fd42712.json) · result
- [MANAGED_SESSION_slow_v1-08b4c9a0.json](MANAGED_SESSION_slow_v1-08b4c9a0.json) · result

</details>

## Related work

- shipped in: [Submission #336334: paid-prefix row-energy correction with exact split-view batching](../../../submissions/336334/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-row-energy-correction-request-risk-25-30-then-23) · [Copy provenance](../../../PROVENANCE.json)

