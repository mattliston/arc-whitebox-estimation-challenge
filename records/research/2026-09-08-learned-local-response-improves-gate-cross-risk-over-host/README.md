# Learned local response improves gate cross-risk over the host: ratios 0.8665908677241692 and 0.8793964725395004

2026-09-08 · finding · informational · measured

Independent-stream products estimated gate-error risk with sampling-MSE bias removed in expectation. The learned response passed the fixed gate-information comparisons on both networks, while shared accelerator arithmetic bias and final covariance or score quality remained unqualified.

Recorded measurement:

    Rank35 learned/host0.8665908677241692,learned/Gaussian0.01827106888122442;rank100 .8793964725395004/.01714432676877074;learned crossMSE2.58738208729921e-8/2.3348715084302143e-8.

## Programs

- [check.py](check.py)
- [score_gate_information.py](score_gate_information.py)
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
<summary>Additional programs and dependencies (16)</summary>

- [../../../experiments/learned-correction/full-fp64-response-shadow-fails-current-width-primal-compatibility/code/current_response_quadratic_v1/feature_second_jet_reference.py](../../../experiments/learned-correction/full-fp64-response-shadow-fails-current-width-primal-compatibility/code/current_response_quadratic_v1/feature_second_jet_reference.py)
- [../../../experiments/learned-correction/full-fp64-response-shadow-fails-current-width-primal-compatibility/code/current_response_quadratic_v1/gnn_second_jet_reference.py](../../../experiments/learned-correction/full-fp64-response-shadow-fails-current-width-primal-compatibility/code/current_response_quadratic_v1/gnn_second_jet_reference.py)
- [feature_native_jet.py](feature_native_jet.py)
- [gnn_native_jet.py](gnn_native_jet.py)
- [native_reference.py](native_reference.py)
- [qualify_native.py](qualify_native.py)
- [diagnose_scalar.py](diagnose_scalar.py)
- [../2026-09-08-exact-copula-endpoints-classify-18-345696-pairs-infeasible-correct/scalar_interval.py](../2026-09-08-exact-copula-endpoints-classify-18-345696-pairs-infeasible-correct/scalar_interval.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_response_rows_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_response_rows_v1.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [result.json#L1](result.json#L1)

<details>
<summary>All result files and records (2)</summary>

- [result.json](result.json) · record
- [result-52938606.json](result-52938606.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-learned-local-response-improves-gate-cross-risk-over-host) · [Copy provenance](../../../PROVENANCE.json)

