# Hermite grouping exceeds the request-risk gate at 26.17%; view caching repairs it to 19.35%

2026-10-15T18:22:26Z · incident · adopted · measured

The first release candidate passed setup and wall limits but exceeded the unchanged request-risk limit. Arena and output-view reuse preserved all predictions and FLOPs on ten native networks and allowed the original private test to transfer.

Recorded measurement:

    first48862/warm46779; risk26.17%>20%; setup4.439s,worst91.3s; proxy504first/536warm requests saved; repaired risk19.35%; private8 adjusted-.639812%

## Programs

- [candidate_hermite_group_views_QR_HELD_v1.py](candidate_hermite_group_views_QR_HELD_v1.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_hermite_group_views_QR_HELD_v2.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_hermite_group_views_QR_HELD_v2.py)
- [fragment_hermite_group_views_HELD_v1.py](fragment_hermite_group_views_HELD_v1.py)
- [check_views_v1.py](check_views_v1.py)
- [check_views_v2.py](check_views_v2.py)
- [finalize_metadata_v1.py](finalize_metadata_v1.py)
- [finalize_metadata_v2.py](finalize_metadata_v2.py)
- [finish_view_release_v1.py](finish_view_release_v1.py)
- [../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

<details>
<summary>Additional programs and dependencies (34)</summary>

- [../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/quick.py](../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/quick.py)
- [../../../instruments/single-attempt-submission-uploader-records-one-qualified-archive-receipt/code/fire_once.py](../../../instruments/single-attempt-submission-uploader-records-one-qualified-archive-receipt/code/fire_once.py)
- [../../../experiments/third-order-planes/read-side-rank-reduced-320-two-network-adjusted-2/code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](../../../experiments/third-order-planes/read-side-rank-reduced-320-two-network-adjusted-2/code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)
- [early_count_gate_v1.py](early_count_gate_v1.py)
- [finite_gop2_current_boundary_HELD_v1.py](finite_gop2_current_boundary_HELD_v1.py)
- [../../../experiments/third-order-planes/read-side-rank-reduced-320-two-network-adjusted-2/code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](../../../experiments/third-order-planes/read-side-rank-reduced-320-two-network-adjusted-2/code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [finite_gop2_current_native10_HELD_v1.py](finite_gop2_current_native10_HELD_v1.py)
- [finite_gop2_current_census_HELD_v1.py](finite_gop2_current_census_HELD_v1.py)
- [../../../experiments/covariance-structure/hermite-covariance-grouping-into-two-products-per-order-all/code/lead/runs/astra_hermite_group_fusion_v1/fragment_hermite_group_HELD_v1.py](../../../experiments/covariance-structure/hermite-covariance-grouping-into-two-products-per-order-all/code/lead/runs/astra_hermite_group_fusion_v1/fragment_hermite_group_HELD_v1.py)
- [../../../experiments/covariance-structure/hermite-covariance-grouping-into-two-products-per-order-all/code/lead/runs/astra_hermite_group_request_lean_v1/candidate_hermite_group_lean_QR_HELD_v2.py](../../../experiments/covariance-structure/hermite-covariance-grouping-into-two-products-per-order-all/code/lead/runs/astra_hermite_group_request_lean_v1/candidate_hermite_group_lean_QR_HELD_v2.py)
- [../../../experiments/covariance-structure/hermite-covariance-grouping-into-two-products-per-order-all/code/lead/runs/astra_hermite_group_request_lean_v1/check_alias_v3.py](../../../experiments/covariance-structure/hermite-covariance-grouping-into-two-products-per-order-all/code/lead/runs/astra_hermite_group_request_lean_v1/check_alias_v3.py)
- [../../../experiments/covariance-structure/hermite-covariance-grouping-into-two-products-per-order-all/code/lead/runs/astra_hermite_group_request_lean_v1/check_tiny_tariff_lean_v2.py](../../../experiments/covariance-structure/hermite-covariance-grouping-into-two-products-per-order-all/code/lead/runs/astra_hermite_group_request_lean_v1/check_tiny_tariff_lean_v2.py)
- [../../../experiments/covariance-structure/hermite-covariance-grouping-into-two-products-per-order-all/code/lead/runs/astra_hermite_group_request_lean_v1/fragment_hermite_group_lean_HELD_v2.py](../../../experiments/covariance-structure/hermite-covariance-grouping-into-two-products-per-order-all/code/lead/runs/astra_hermite_group_request_lean_v1/fragment_hermite_group_lean_HELD_v2.py)
- [finite_qhfl_seal_arms_v1.py](finite_qhfl_seal_arms_v1.py)
- [grade_transferred_qhfl_seal8_v1.py](grade_transferred_qhfl_seal8_v1.py)
- [finite_seal_four_partition_private_v2.py](finite_seal_four_partition_private_v2.py)
- [run_cert_private_truth_HELD_v1.py](run_cert_private_truth_HELD_v1.py)
- [../../../experiments/third-order-planes/read-prefix-cutoff-reduced-1-5-1-0-standard/code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py](../../../experiments/third-order-planes/read-prefix-cutoff-reduced-1-5-1-0-standard/code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_rtt2_request_repair_HELD_v2.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_rtt2_request_repair_HELD_v2.py)
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
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/stage_c_boundary.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/stage_c_boundary.py)
- [../../../experiments/arithmetic-kernels/merged-newborn-common-product-parked-warm-flops-0-148/code/scout_native/run_cert_v2.py](../../../experiments/arithmetic-kernels/merged-newborn-common-product-parked-warm-flops-0-148/code/scout_native/run_cert_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3466](../../TEAM_LOG_2026-10-01_to_10-22.md#L3466)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3467](../../TEAM_LOG_2026-10-01_to_10-22.md#L3467)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3469](../../TEAM_LOG_2026-10-01_to_10-22.md#L3469)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3470](../../TEAM_LOG_2026-10-01_to_10-22.md#L3470)

<details>
<summary>All result files and records (16)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SOURCE_BUILD_v1.json](SOURCE_BUILD_v1.json) · result
- [SOURCE_BUILD_v2.json](SOURCE_BUILD_v2.json) · result
- [SOURCE_READY_v1.json](SOURCE_READY_v1.json) · result
- [SOURCE_READY_v2.json](SOURCE_READY_v2.json) · result
- [TINY_VIEWS_AND_REQUESTS_v1.json](TINY_VIEWS_AND_REQUESTS_v1.json) · result
- [TINY_VIEWS_AND_REQUESTS_v2.json](TINY_VIEWS_AND_REQUESTS_v2.json) · result
- [RELEASE_CONFIRMED_v1.json](RELEASE_CONFIRMED_v1.json) · result
- [ACTUAL_MANAGED_EXIT_normal_v1.json](ACTUAL_MANAGED_EXIT_normal_v1.json) · result
- [ACTUAL_MANAGED_EXIT_slow_v1.json](ACTUAL_MANAGED_EXIT_slow_v1.json) · result
- [DELIVERY_READY_v1.json](DELIVERY_READY_v1.json) · result
- [FINALIZATION_COPY_01_v1.json](FINALIZATION_COPY_01_v1.json) · result
- [FINALIZATION_COPY_03_v1.json](FINALIZATION_COPY_03_v1.json) · result
- [FINALIZATION_SEQUENCE_HELD_v1.json](FINALIZATION_SEQUENCE_HELD_v1.json) · result
- [MANAGED_SESSION_normal_v1.json](MANAGED_SESSION_normal_v1.json) · result
- [MANAGED_SESSION_slow_v1.json](MANAGED_SESSION_slow_v1.json) · result

</details>

## Related work

- shipped in: [Submission #336294: Hermite covariance grouping and QR read with exact view caching](../../../submissions/336294/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-hermite-grouping-exceeds-request-risk-gate-26-17-view) · [Copy provenance](../../../PROVENANCE.json)

