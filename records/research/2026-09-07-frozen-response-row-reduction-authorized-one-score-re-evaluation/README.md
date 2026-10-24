# Frozen response-row reduction authorized for one score re-evaluation with four-level Winograd products

2026-09-07T02:28Z · decision · adopted · measured

After the owner's explicit narrow grant, the previously raw-only-closed learned response-row reduction (frozen payload P with 16 full-feature heads and 64 recurrent channels, response selection threshold .01) was ported without any fit into the current bank runtime (four-level Winograd product kernel D4, tile 8192, bulk copies) as candidate_response_rows_current_runtime_v1.py (SHA 5a07832f...), with a preregistered score-aware gate: a lower two-net mean of per-net raw x max(0.1, complete FLOPs/2^41) on ranks 35/100 advances to the independent 8-public + 6-virgin panel, anything else closes the exception. The source port and the fixed evaluator were accepted after full reads before execution.

Recorded measurement:

    Historical raw change of the learned response rows vs its matched parent: +3.1204% (rank 35), -0.3523% (rank 100); planning range 0.47-0.52T FLOPs per MLP vs about 0.49T for the bank.

## Programs

- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py)
- [../../../submissions/329851/estimator.py](../../../submissions/329851/estimator.py)
- [../../../submissions/329926/estimator.py](../../../submissions/329926/estimator.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_response_rows_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_response_rows_v1.py)

</details>

## Evidence

- [RESPONSE_ROWS_ADJUSTED_EXCEPTION_V1.md#L3](RESPONSE_ROWS_ADJUSTED_EXCEPTION_V1.md#L3)
- [RESPONSE_ROWS_ADJUSTED_EXCEPTION_V1.md#L93](RESPONSE_ROWS_ADJUSTED_EXCEPTION_V1.md#L93)
- [RESPONSE_ROWS_CURRENT_RUNTIME_SOURCE_ACCEPTANCE_V1.md#L11](RESPONSE_ROWS_CURRENT_RUNTIME_SOURCE_ACCEPTANCE_V1.md#L11)

<details>
<summary>All result files and records (3)</summary>

- [RESPONSE_ROWS_ADJUSTED_EXCEPTION_V1.md](RESPONSE_ROWS_ADJUSTED_EXCEPTION_V1.md) · record
- [RESPONSE_ROWS_CURRENT_RUNTIME_SOURCE_ACCEPTANCE_V1.md](RESPONSE_ROWS_CURRENT_RUNTIME_SOURCE_ACCEPTANCE_V1.md) · record
- [RESPONSE_ROWS_ADJUSTED_CHECKER_ACCEPTANCE_V1.md](RESPONSE_ROWS_ADJUSTED_CHECKER_ACCEPTANCE_V1.md) · record

</details>

## Related work

- ← enables: [Frozen response-row reduction improves two-network adjusted score by 3.63221%](../../../experiments/learned-correction/frozen-response-row-reduction-improves-two-network-adjusted-score/README.md)

[Research graph](../../../graph/index.html#node=2026-09-07-frozen-response-row-reduction-authorized-one-score-re-evaluation) · [Copy provenance](../../../PROVENANCE.json)

