# Submission #328210: full-budget closure with order-24 pair series, raw 5.765125013112993e-7

2026-08-24T10:36:50Z · submission · adopted · measured

The frozen order-24 Hermite pair-series closure (Q24) estimator (b38cd865: the full-budget closure with the cap-16384 third-cumulant canonical-polyadic carrier (CP3) bank, fused third-cumulant repeated-index plane (K21) plane reads, coherent quartic action with the cap-2048 fourth-cumulant canonical-polyadic carrier (CP4) carrier, spectral positive-semidefinite (PSD) repair and the pair Edgeworth consumer, with the Hermite/rho pair series cut from q=0..32 to q=0..24) consumed its sole preregistered attempt and graded with a 100/100 operational pass and zero failed rows. Its raw final-layer mean squared error (MSE) is 57.65 times the 1e-8 target, so it failed promotion; five of the 50 displayed public rows above 5e-7 carried 86.7525% of the displayed raw sum, which localised a tail for follow-up. The ledger marks it historical and not the Phase 2 baseline.

Recorded measurement:

    score_adjusted_exact 5.587487766740284e-07, raw final 5.765125013112993e-07, all_layers_mse_display 8.232e-08, sampling_mse_display 1.133e-06, best MLP adjusted 5.142e-08, worst 1.452e-05; 50 public rows median raw 8.5555e-8; bill 2,131,249,329,109 FLOPs (96.9180% of 2^41), wall 29.222195 s, residual 0.294491 s, RSS 4,507,058,176 bytes; evaluation 10:39:41Z to 11:25:43Z, smoke 12/12

Public score: 5.587487766740284e-07. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../../experiments/flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/compare_target_free_v4_q24.py](../../experiments/flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/compare_target_free_v4_q24.py)
- [../../experiments/flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/run_additional_fullwidth_seeds.py](../../experiments/flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/run_additional_fullwidth_seeds.py)
- [../328199/estimator.py](../328199/estimator.py)
- [../../instruments/literal-meter-harness-subprocess-evaluation-run-records-verification-resource/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py](../../instruments/literal-meter-harness-subprocess-evaluation-run-records-verification-resource/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py)
- [../../instruments/literal-meter-harness-subprocess-evaluation-run-records-verification-resource/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py](../../instruments/literal-meter-harness-subprocess-evaluation-run-records-verification-resource/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py)

## Evidence

- [SUBMISSION_LEDGER.md#L671](../../records/SUBMISSION_LEDGER.md#L671)
- [SUBMISSION_LEDGER.md#L674](../../records/SUBMISSION_LEDGER.md#L674)
- [BOARD_ATTEMPT_328210_RESULT.json#L1](../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/BOARD_ATTEMPT_328210_RESULT.json#L1)

<details>
<summary>All result files and records (5)</summary>

- [BOARD_ATTEMPT_328210_RESULT.json](../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/BOARD_ATTEMPT_328210_RESULT.json) · record
- [INDEPENDENT_BOARD_RESULT_AUDIT_328210.json](../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/INDEPENDENT_BOARD_RESULT_AUDIT_328210.json) · record
- [FREEZE_RECEIPT_Q24.json](../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/FREEZE_RECEIPT_Q24.json) · record
- [receipt-d69fd338.json](../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/receipt-d69fd338.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- ← enables: [Submission #328199: full-budget closure with order-32 pair series exceeded the 120 s wall](../328199/README.md)
- shipped in: [Pair-series order 32 reduced to 24: final prediction disagreement at most 9.45e-17 on three full-width seeds](../../experiments/wall-and-residual/pair-series-order-32-reduced-24-final-prediction-disagreement/README.md)
- ← supersedes: [Submission #328273: order-24 closure with hybrid variance-floor repair, raw 9.232750883825246e-8](../328273/README.md)

[Research graph](../../graph/index.html#node=submission-328210) · [Copy provenance](../../PROVENANCE.json)

