# Combined-candidate workflow applies verified changes, seals on fresh networks and runs a fast release gate

2026-10-12T17:28Z · decision · informational · measured

Matt asked for consistent stackable wins; two scout teammates (moment-closure estimator physics, native metered FLOPs) delivered ranked backlogs with kill tests, the engineer's stack builder applied literal / fragment deltas to the latest round's files, a pre-registered seal on 8 fresh seal-block networks (submission iff mean < 0, extend to 16 between 0 and +1 se) and release's fast gate re-assessed the engineer's own count sessions.

Recorded measurement:

    acceptance rule from the podium curve: fire iff S_mean < 0.12 x sd^2; seal block 7190000-031 (8M) + 7190032-047 (16M); door to door 70-100 min

## Programs

- [stack_build_v2.sh](stack_build_v2.sh)
- [fastgate_v1.sh](fastgate_v1.sh)
- [fastgate_v5.sh](fastgate_v5.sh)
- [fastgate_loop_v1.py](fastgate_loop_v1.py)
- [../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/engineer/a41_base/fragment_gsk_v1.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/engineer/a41_base/fragment_gsk_v1.py)
- [fastgate_post_v1.py](fastgate_post_v1.py)
- [fragment_gskv_gk11.py](fragment_gskv_gk11.py)
- [../2026-10-11-six-native-jobs-trip-swap-guard-five-terminate-together/gate_rdz_v1.py](../2026-10-11-six-native-jobs-trip-swap-guard-five-terminate-together/gate_rdz_v1.py)

<details>
<summary>Additional programs and dependencies (25)</summary>

- [../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_gsk_v1.py](../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_gsk_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py)
- [../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_xsg_any_v1.py](../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_xsg_any_v1.py)
- [../../../instruments/targeted-prebinding-replaces-unused-keys-first-network-requests-fall/code/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py](../../../instruments/targeted-prebinding-replaces-unused-keys-first-network-requests-fall/code/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py)
- [candidate.py](candidate.py)
- [candidate-fd58b9e2.py](candidate-fd58b9e2.py)
- [../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py](../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py)
- [seal_judge_v1.py](seal_judge_v1.py)
- [stack_deltas_v2.py](stack_deltas_v2.py)
- [../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/quick.py](../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/quick.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [fastgate_pin_v1.py](fastgate_pin_v1.py)
- [fastgate_v2.sh](fastgate_v2.sh)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/release/guard_footprint_v2.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/release/guard_footprint_v2.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [compare_twin_byteid_v1.py](compare_twin_byteid_v1.py)
- [compare_twin_native_v3.py](compare_twin_native_v3.py)
- [compare_twin_native_v4.py](compare_twin_native_v4.py)
- [compare_twin_native_v5.py](compare_twin_native_v5.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py)
- [../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py](../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1558](../../TEAM_LOG_2026-10-01_to_10-22.md#L1558)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1561](../../TEAM_LOG_2026-10-01_to_10-22.md#L1561)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1567](../../TEAM_LOG_2026-10-01_to_10-22.md#L1567)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1570](../../TEAM_LOG_2026-10-01_to_10-22.md#L1570)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1575](../../TEAM_LOG_2026-10-01_to_10-22.md#L1575)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1582](../../TEAM_LOG_2026-10-01_to_10-22.md#L1582)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1584](../../TEAM_LOG_2026-10-01_to_10-22.md#L1584)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1589](../../TEAM_LOG_2026-10-01_to_10-22.md#L1589)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1590](../../TEAM_LOG_2026-10-01_to_10-22.md#L1590)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1593](../../TEAM_LOG_2026-10-01_to_10-22.md#L1593)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2335](../../TEAM_LOG_2026-10-01_to_10-22.md#L2335)

<details>
<summary>All result files and records (4)</summary>

- [seal_s51k1_PREREG.json](seal_s51k1_PREREG.json) · record
- [stack_deltas_gk11_v1.json](stack_deltas_gk11_v1.json) · record
- [seal_block_v2.json](seal_block_v2.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Combined candidates receive one sealed comparison against the submitted estimator before promotion](../2026-10-11-combined-candidates-receive-one-sealed-comparison-against-submitted-estimator/README.md)
- enables: [Setup prebind guard makes local FLOP counts load-dependent; release uses deterministic count sessions](../2026-10-12-setup-prebind-guard-makes-local-flop-counts-load-dependent/README.md)
- enables: [Skew- and kurtosis-corrected transport gate adopted: -0.348% pooled MSE on 24 fresh networks](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/README.md)
- enables: [Only combined candidates receive release qualification after accepted refit rounds](../2026-10-13-only-combined-candidates-receive-release-qualification-after-accepted-refit/README.md)
- enables: [Submit when accumulated validated net improvement reaches 0.2% and release checks pass](../2026-10-13-submit-when-accumulated-validated-net-improvement-reaches-0-2/README.md)
- ← enables: [Generated refit-after-drop rounds connect sizing, porting, sealed judging and release through verdict files](../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks) · [Copy provenance](../../../PROVENANCE.json)

