# Scalar interval arithmetic explains finite-difference failure; the original whole certificate remains failed

2026-09-08 · finding · informational · measured

A fixed high-precision scalar reference certified the first and second automatic derivatives and explained the earlier coarse finite-difference discrepancy. This was a local reference diagnosis, leaving the original full numerical qualification failed and without opening gate truth.

Recorded measurement:

    160-digit directed Decimal contexts;512CDF series terms;ad_accuracy_passed=true;original_qualification_remains_failed=true;wall0.419853875413537s.

## Programs

- [../2026-09-08-learned-local-response-improves-gate-cross-risk-over-host/diagnose_scalar.py](../2026-09-08-learned-local-response-improves-gate-cross-risk-over-host/diagnose_scalar.py)
- [../2026-09-08-exact-copula-endpoints-classify-18-345696-pairs-infeasible-correct/scalar_interval.py](../2026-09-08-exact-copula-endpoints-classify-18-345696-pairs-infeasible-correct/scalar_interval.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py)
- [../../../submissions/329851/estimator.py](../../../submissions/329851/estimator.py)
- [../../../submissions/329926/estimator.py](../../../submissions/329926/estimator.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [../../../experiments/learned-correction/full-fp64-response-shadow-fails-current-width-primal-compatibility/code/current_response_quadratic_v1/feature_second_jet_reference.py](../../../experiments/learned-correction/full-fp64-response-shadow-fails-current-width-primal-compatibility/code/current_response_quadratic_v1/feature_second_jet_reference.py)
- [../2026-09-08-learned-local-response-improves-gate-cross-risk-over-host/feature_native_jet.py](../2026-09-08-learned-local-response-improves-gate-cross-risk-over-host/feature_native_jet.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_response_rows_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_response_rows_v1.py)

</details>

## Evidence

- [result.json#L2](result.json#L2)

<details>
<summary>All result files and records (1)</summary>

- [result.json](result.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-scalar-interval-arithmetic-explains-finite-difference-failure-original-whole) · [Copy provenance](../../../PROVENANCE.json)

